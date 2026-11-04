# ruff: noqa: E501
import ctypes
from dataclasses import dataclass
import functools
import hashlib
import itertools
import math
import os
import pathlib
import re
import tempfile
import time
from typing import Any

from tinygrad.runtime.autogen import mesa
from tinygrad.codegen import to_program
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.engine.realize import get_runtime
from tinygrad.engine.worker import get_worker_pool
from tinygrad.helpers import DEBUG, Context, colored, diskcache_get, diskcache_put
from tinygrad.runtime.autogen import libc
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import dtypes

NUM_R_REG = 48 * 4
# per-lane storage: the 48 full vec4 registers, then room for a separate half-register file. SP_CS_CNTL_0.MERGEDREGS selects whether
# hrN.c aliases the halves of the full registers (merged) or lives in its own file (not merged); tinygrad leaves the bit clear.
R_WORDS = NUM_R_REG * 2
NUM_S_REG = 8 * 4
NUM_C_REG = 512 * 4
NUM_A_REG = 2 * 1
NUM_P_REG = 1 * 4
NUM_IMG_DESC = 32  # texture / IBO descriptors visible to a shader

def _c(val, dtype=dtypes.uint32): return UOp.const(val, dtype)


def _as_float32(value: UOp) -> UOp:
    """Interpret a 32-bit register payload as IEEE-754 float32."""
    if value.dtype == dtypes.float32:
        return value
    if value.dtype.itemsize == dtypes.float32.itemsize:
        return value.bitcast(dtypes.float32)
    return value.cast(dtypes.float32)


def _as_reg_bits(value: UOp) -> UOp:
    """Encode a scalar result as the raw bits stored in a 32-bit IR3 register component."""
    if value.dtype == dtypes.float32: return value.bitcast(dtypes.uint32)
    if value.dtype == dtypes.float16: return value.bitcast(dtypes.uint16).cast(dtypes.uint32)
    return value.cast(dtypes.uint32)


def _as_float16(value: UOp) -> UOp:
    """Decode a register's low 16 bits as binary16, or convert a typed float immediate."""
    if value.dtype == dtypes.float16: return value.cast(dtypes.float32)
    # Textual float immediates are already numerical values; register operands are raw bits.
    if value.dtype == dtypes.float32: return value.cast(dtypes.float16).cast(dtypes.float32)
    return value.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)


def _qcom_demoted_const_f16(ctx, token: str) -> UOp:
    """A half-precision float read of a constant: the 32-bit constant is converted f32 -> f16 (CONSTANT_DEMOTION_ENABLE)."""
    _, reg, dim = _qcom_reg_token(token)
    return ctx.read_c_reg(_c(reg, dtypes.int), _c(dim, dtypes.int)).bitcast(dtypes.float32).cast(dtypes.float16).cast(dtypes.float32)


def _qcom_float_operand(ctx, field, lane: UOp) -> UOp:
    """Read an operand as a float before applying its typed abs/neg modifiers."""
    raw, mods = _qcom_read_raw_operand(ctx, field.value, lane, field.modifiers)
    try:
        kind, _, _ = _qcom_reg_token(field.value)
    except ValueError:
        kind = ""
    if kind == "hc": value = _qcom_demoted_const_f16(ctx, field.value)
    else: value = _as_float16(raw) if kind in {"hr", "hs"} else _as_float32(raw)
    return _apply_qcom_operand_modifiers(value, mods)


def _qcom_write_result(ctx, token: str, lane: UOp, value: UOp):
    """Write a typed result using the destination register's raw-bit conventions."""
    return UOp.sink(_write_qcom_store(ctx, token, lane, value))


def _qcom_apply_instruction_flags(inst: "Inst", value: UOp) -> UOp:
    """Apply supported result flags encoded before the mnemonic in the disassembly."""
    if not inst.fields or "sat" not in inst.fields[0].modifiers:
        return value
    if value.dtype not in {dtypes.float16, dtypes.float32}:
        raise NotImplementedError(f"(sat) on {_inst_name(inst)!r} with {value.dtype} result is not implemented")
    zero, one = UOp.const(0.0, value.dtype), UOp.const(1.0, value.dtype)
    return value.maximum(zero).minimum(one)

def _thread_active(exec_mask: UOp, thread_id: UOp) -> UOp:
  if exec_mask.dtype == dtypes.uint64: return ((exec_mask >> thread_id.cast(dtypes.uint64)) & UOp.const(1, dtypes.uint64)).ne(UOp.const(0, dtypes.uint64))
  return ((exec_mask >> thread_id.cast(dtypes.uint32)) & _c(1)).ne(_c(0))

class _Ctx:
    __slots__ = ("_axis_id", "wave_size", "merged", "r_reg", "s_reg", "c_reg", "a_reg", "p_reg", "vmem", "shared", "pvt_mem", "tmp", "tex", "ibo", "_exec_mask", "_bufs", "fused_lane")
    # The kernel's buffers, in argument order. Half registers are packed into the low/high halves of the full-width register buffer.
    r_reg: UOp
    s_reg: UOp
    c_reg: UOp
    a_reg: UOp
    p_reg: UOp
    vmem: UOp
    shared: UOp
    pvt_mem: UOp
    tmp: UOp
    tex: UOp
    ibo: UOp
    _exec_mask: UOp

    def __init__(self, wave_size = 64, merged = False):
        self._axis_id = 0
        self.wave_size = wave_size
        self.merged = merged
        self._bufs = {
            "r_reg": UOp.param(0, dtypes.uint32, R_WORDS * wave_size, name="r_reg"), "s_reg": UOp.param(1, dtypes.uint32, NUM_S_REG, name="s_reg"),
            "c_reg": UOp.param(2, dtypes.uint32, NUM_C_REG, name="c_reg"), "a_reg": UOp.param(3, dtypes.uint32, NUM_A_REG, name="a_reg"),
            "p_reg": UOp.param(4, dtypes.uint32, NUM_P_REG * wave_size, name="p_reg"),
            "vmem": UOp.param(5, dtypes.uint8, 1<<48, name="vmem"),  # host VAs reach 2**47
            "shared": UOp.param(6, dtypes.uint8, 32 * 1024, name="shared"), "pvt_mem": UOp.param(7, dtypes.uint8, 1 << 30, name="pvt_mem"),
            "tmp": UOp.param(9, dtypes.uint32, 4 * wave_size, name="tmp")}  # scratch: swz/gat/sct park their sources here
        self._exec_mask = UOp.param(8, dtypes.uint64, 1, name="exec_mask")
        self.fused_lane: UOp | None = None
        self.tex = UOp.param(10, dtypes.uint32, 16 * NUM_IMG_DESC, name="tex")  # 16-dword texture / IBO descriptors (A6XX_TEX_CONST)
        self.ibo = UOp.param(11, dtypes.uint32, 16 * NUM_IMG_DESC, name="ibo")
        for name, buf in self._bufs.items(): setattr(self, name, buf)

    def advance(self, done: UOp) -> None:
        """Make every later access to a buffer wait for `done` (a finished instruction), so instructions can share one kernel."""
        for name, buf in self._bufs.items(): setattr(self, name, buf.after(done))

    def range(self, n: int | None = None) -> UOp:
        if self.fused_lane is not None: return self.fused_lane  # instructions fused into one lane loop share its range
        if n is None: n = self.wave_size
        self._axis_id += 1
        return UOp.range(n, self._axis_id, dtype=dtypes.int)

    def read_exec_mask(self) -> UOp:
        return self._exec_mask.index(_c(0, dtypes.int)).load()

    def read_r_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, half=False):
        reg_slot = reg_num.cast(dtypes.int) * _c(4, dtypes.int) + reg_dim.cast(dtypes.int)
        if half:
            # The half-register view is the uint16 view of the same per-lane uint32 file.
            half_idx = thread_id.cast(dtypes.int) * _c(R_WORDS * 2, dtypes.int) + reg_slot + _c(0 if self.merged else NUM_R_REG * 2, dtypes.int)
            word_idx = half_idx // _c(2, dtypes.int)
            half_shift = (half_idx & _c(1, dtypes.int)).cast(dtypes.uint32) * _c(16, dtypes.uint32)
            word = self.r_reg.index(word_idx).load()
            return ((word >> half_shift) & _c(0xFFFF, dtypes.uint32)).cast(dtypes.uint32)
        idx = thread_id.cast(dtypes.int) * _c(R_WORDS, dtypes.int) + reg_slot
        return self.r_reg.index(idx).load()

    #TODO: check all the exec masks and afters
    #may be better to get rid of afters, there should be no ordering problems
    def write_r_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None, half=False):
        buf = self.r_reg.after(after) if after is not None else self.r_reg
        reg_slot = reg_num.cast(dtypes.int) * _c(4, dtypes.int) + reg_dim.cast(dtypes.int)
        active = _thread_active(exec_mask, thread_id)
        if half:
            half_idx = thread_id.cast(dtypes.int) * _c(R_WORDS * 2, dtypes.int) + reg_slot + _c(0 if self.merged else NUM_R_REG * 2, dtypes.int)
            word_idx = half_idx // _c(2, dtypes.int)
            half_shift = (half_idx & _c(1, dtypes.int)).cast(dtypes.uint32) * _c(16, dtypes.uint32)
            old = buf.index(word_idx).load()
            mask = _c(0xFFFF, dtypes.uint32) << half_shift
            merged = (old & ~mask) | ((val.cast(dtypes.uint32) & _c(0xFFFF, dtypes.uint32)) << half_shift)
            return buf.index(word_idx.valid(active)).store(merged)
        idx = thread_id.cast(dtypes.int) * _c(R_WORDS, dtypes.int) + reg_slot
        return buf.index(idx.valid(active)).store(val.cast(dtypes.uint32))

    def read_s_reg(self, reg_num: UOp, reg_dim: UOp, half=False):
        slot = reg_num.cast(dtypes.int) * _c(4, dtypes.int) + reg_dim.cast(dtypes.int)
        idx = slot // _c(2, dtypes.int) if half else slot
        word = self.s_reg.index(idx).load()
        if not half: return word
        shift = (slot & _c(1, dtypes.int)).cast(dtypes.uint32) * _c(16, dtypes.uint32)
        return (word >> shift) & _c(0xFFFF, dtypes.uint32)

    def write_s_reg(self, reg_num: UOp, reg_dim: UOp, val: UOp, after: UOp | None = None, half=False):
        buf = self.s_reg.after(after) if after is not None else self.s_reg
        slot = reg_num.cast(dtypes.int) * _c(4, dtypes.int) + reg_dim.cast(dtypes.int)
        if half:
            idx = slot // _c(2, dtypes.int)
            shift = (slot & _c(1, dtypes.int)).cast(dtypes.uint32) * _c(16, dtypes.uint32)
            mask = _c(0xFFFF, dtypes.uint32) << shift
            old = buf.index(idx).load()
            merged = (old & ~mask) | ((val.cast(dtypes.uint32) & _c(0xFFFF, dtypes.uint32)) << shift)
            return buf.index(idx).store(merged)
        return buf.index(slot).store(val.cast(dtypes.uint32))

    def read_c_reg(self, reg_num: UOp, reg_dim: UOp, half=False):
        slot = reg_num.cast(dtypes.int) * _c(4, dtypes.int) + reg_dim.cast(dtypes.int)
        idx = slot // _c(2, dtypes.int) if half else slot
        word = self.c_reg.index(idx).load()
        if not half: return word
        shift = (slot & _c(1, dtypes.int)).cast(dtypes.uint32) * _c(16, dtypes.uint32)
        return (word >> shift) & _c(0xFFFF, dtypes.uint32)

    def write_c_reg(self, reg_num: UOp, reg_dim: UOp, val: UOp, after: UOp | None = None, half=False):
        buf = self.c_reg.after(after) if after is not None else self.c_reg
        slot = reg_num.cast(dtypes.int) * _c(4, dtypes.int) + reg_dim.cast(dtypes.int)
        if half:
            idx = slot // _c(2, dtypes.int)
            shift = (slot & _c(1, dtypes.int)).cast(dtypes.uint32) * _c(16, dtypes.uint32)
            mask = _c(0xFFFF, dtypes.uint32) << shift
            old = buf.index(idx).load()
            merged = (old & ~mask) | ((val.cast(dtypes.uint32) & _c(0xFFFF, dtypes.uint32)) << shift)
            return buf.index(idx).store(merged)
        return buf.index(slot).store(val.cast(dtypes.uint32))

    def read_a_reg(self, reg_num: UOp): #TODO: make sure dim is actually unnecessary
        return self.a_reg.index(reg_num.cast(dtypes.int)).load()

    def write_a_reg(self, reg_num: UOp, val: UOp, after: UOp | None = None):
        buf = self.a_reg.after(after) if after is not None else self.a_reg
        return buf.index(reg_num.cast(dtypes.int)).store(val.cast(dtypes.uint32))

    def read_p_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp):
        # UOp backing stores one 0/1 value per lane; WaveState packs it to a mask
        # only when the Python scheduler needs to make a wave-uniform decision.
        slot = reg_num.cast(dtypes.int) * _c(4, dtypes.int) + reg_dim.cast(dtypes.int)
        idx = thread_id.cast(dtypes.int) * _c(NUM_P_REG, dtypes.int) + slot
        return self.p_reg.index(idx).load()

    def write_p_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, val: UOp,
                    exec_mask: UOp, after: UOp | None = None):
        buf = self.p_reg.after(after) if after is not None else self.p_reg
        slot = reg_num.cast(dtypes.int) * _c(4, dtypes.int) + reg_dim.cast(dtypes.int)
        idx = thread_id.cast(dtypes.int) * _c(NUM_P_REG, dtypes.int) + slot
        active = _thread_active(exec_mask, thread_id)
        predicate = val.ne(_c(0, val.dtype)).cast(dtypes.uint32)
        return buf.index(idx.valid(active)).store(predicate)

    def mem_store_reg(self, mem: UOp, addr: UOp, val: UOp):
        # TODO: implement packed register stores when the emulator gets real memory semantics.
        return mem.index(addr).store(val.cast(dtypes.uint32))

    def mem_store_word(self, mem: UOp, addr: UOp, val: UOp):
        idx = mem.index(addr)
        return idx.store(val.cast(dtypes.uint32))

    def mem_store_byte(self, mem: UOp, addr: UOp, val: UOp):
        idx = mem.index(addr)
        return idx.store(val.cast(dtypes.uint8))


def _compile_cat0(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 0 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    # These instructions have their effects in the Python wave scheduler. They need a
    # no-op runner only so the instruction stream and runtime cache remain uniform.
    if name in {"nop", "end", "predt", "predf", "prede"}: return UOp.sink()
    # Do not accidentally turn control flow / lane termination into successful no-ops.
    if name in {"jump", "call", "ret", "br", "bany", "ball", "brao", "braa", "brax", "bkt", "kill", "emit", "cut", "chmask", "chsh", "flow_rev", "shpe", "getlast", "getone", "shps", "brac"}:
        raise NotImplementedError(f"Category 0 instruction {name!r} must be handled by the wave scheduler or needs unsupported subgroup/geometry semantics")
    raise RuntimeError(f"Unknown category 0 instruction: {name!r}")


def _write_qcom_store(ctx: _Ctx, token: str, lane: UOp, value: UOp, after: UOp | None = None):
    kind, reg, dim = _qcom_reg_token(token)
    if kind in {"r", "hr"}:
        exec_mask = ctx.read_exec_mask()
        if kind == "hr":
            if value.dtype == dtypes.float32: value = value.cast(dtypes.float16).bitcast(dtypes.uint16)
            elif value.dtype == dtypes.float16: value = value.bitcast(dtypes.uint16)
            else: value = value.cast(dtypes.uint16)
            bits = value.cast(dtypes.uint32)
        else:
            bits = _as_reg_bits(value)
        return ctx.write_r_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane, bits, exec_mask, after=after, half=(kind == "hr"))
    if kind in {"s", "hs"}:
        if kind == "hs":
            if value.dtype == dtypes.float32: value = value.cast(dtypes.float16).bitcast(dtypes.uint16)
            elif value.dtype == dtypes.float16: value = value.bitcast(dtypes.uint16)
            else: value = value.cast(dtypes.uint16)
        return ctx.write_s_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), _as_reg_bits(value), after=after, half=(kind == "hs"))
    if kind in {"c", "hc"}:
        if kind == "hc":
            if value.dtype == dtypes.float32: value = value.cast(dtypes.float16).bitcast(dtypes.uint16)
            elif value.dtype == dtypes.float16: value = value.bitcast(dtypes.uint16)
            else: value = value.cast(dtypes.uint16)
        return ctx.write_c_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), _as_reg_bits(value), after=after, half=(kind == "hc"))
    if kind == "a":
        return ctx.write_a_reg(_c(reg, dtypes.int), _as_reg_bits(value), after=after)
    if kind == "p":
        exec_mask = ctx.read_exec_mask()
        return ctx.write_p_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane, value, exec_mask, after=after)
    raise ValueError(f"Unsupported QCOM register kind {kind!r} in {token!r}")


def _write_qcom_value(ctx: _Ctx, token: str, lane: UOp, value: UOp):
    return UOp.sink(_write_qcom_store(ctx, token, lane, value))


def _qcom_reg_offset_token(token: str, delta: int) -> str:
    kind, reg, dim = _qcom_reg_token(token)
    dim_name = {0: "x", 1: "y", 2: "z", 3: "w"}[dim]
    return f"{kind}{reg + delta}.{dim_name}"


def _qcom_reg_component_offset_token(token: str, delta: int) -> str:
    """Advance a vector operand by components, carrying w into the next register's x."""
    kind, reg, dim = _qcom_reg_token(token)
    if kind not in {"r", "hr", "c", "hc", "s", "hs"}:
        raise NotImplementedError(f"Multi-component accesses require r/c/s registers, got {token!r}")
    slot = reg * 4 + dim + delta
    dim_name = {0: "x", 1: "y", 2: "z", 3: "w"}[slot % 4]
    return f"{kind}{slot // 4}.{dim_name}"


_QCOM_VALUE_TYPES = ("u8_32", "f16", "f32", "u16", "u32", "s16", "s32", "u8")


def _qcom_split_type_pair(suffix: str) -> tuple[str, str] | None:
    for src_type in sorted(_QCOM_VALUE_TYPES, key=len, reverse=True):
        if not suffix.startswith(src_type): continue
        dst_suffix = suffix[len(src_type):]
        if dst_suffix in _QCOM_VALUE_TYPES: return src_type, dst_suffix
    return None


def _qcom_decode_typed(value: UOp, typ: str) -> UOp:
    if typ == "f16": return _as_float16(value)
    if typ == "f32": return _as_float32(value)
    if typ == "u8": return value.cast(dtypes.uint8)
    if typ == "u8_32": return value.cast(dtypes.uint8).cast(dtypes.uint32)
    if typ == "u16": return value.cast(dtypes.uint16)
    if typ == "u32": return value.cast(dtypes.uint32)
    if typ == "s16": return value.cast(dtypes.uint16).bitcast(dtypes.int16)
    if typ == "s32": return value.cast(dtypes.uint32).bitcast(dtypes.int32)
    raise ValueError(f"Unsupported IR3 scalar type {typ!r}")


def _qcom_encode_typed(value: UOp, typ: str) -> UOp:
    if typ == "f16": return value.cast(dtypes.float16)
    if typ == "f32": return value.cast(dtypes.float32)
    if typ == "u8": return value.cast(dtypes.uint8)
    if typ == "u8_32": return value.cast(dtypes.uint8).cast(dtypes.uint32)
    if typ == "u16": return value.cast(dtypes.uint16)
    if typ == "u32": return value.cast(dtypes.uint32)
    if typ == "s16": return value.cast(dtypes.int16)
    if typ == "s32": return value.cast(dtypes.int32)
    raise ValueError(f"Unsupported IR3 scalar type {typ!r}")


def _qcom_u8_source_extends(src: UOp, src_type: str, dst_type: str) -> UOp:
    """An 8-bit cov/mov source sign-extends: ir3_compiler_nir.c uses `cov.u8s16` for i2i16/i2f of int8 and masks with 0xff wherever it
    needs zero extension ("Zero-extension of 8-bit values doesn't work with `cov`")."""
    return src.bitcast(dtypes.int8) if src_type == "u8" and dst_type != "u8" else src


def _compile_cat1(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 1 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    op = name.split(".", 1)[0]
    lane = ctx.range()

    if op in {"swz", "gat", "sct"}:
        if op == "swz":
            if len(inst.fields) != 5: raise RuntimeError(f"swz expects 2 destinations and 2 sources: {inst!r}")
            dsts, srcs = [f.value for f in inst.fields[1:3]], inst.fields[3:5]
            vals = [_qcom_read_operand(ctx, f.value, lane, f.modifiers) for f in srcs]
            targets = dsts
        elif op == "gat":
            # Mesa ISA doc: "Move SRC0 to DST0, SRC1 to DST0 + 1, SRC2 to DST0 + 2, and SRC3 to DST0 + 3" (consecutive components)
            if len(inst.fields) != 6: raise RuntimeError(f"gat expects 1 destination and 4 sources: {inst!r}")
            vals = [_qcom_read_operand(ctx, f.value, lane, f.modifiers) for f in inst.fields[2:6]]
            targets = [_qcom_reg_component_offset_token(inst.fields[1].value, i) for i in range(4)]
        else:
            # Mesa ISA doc: "Move SRC0 to DST0, SRC0 + 1 to DST1, SRC0 + 2 to DST2, and SRC0 + 3 to DST3"
            if len(inst.fields) != 6: raise RuntimeError(f"sct expects 4 destinations and 1 source: {inst!r}")
            src = inst.fields[5]
            vals = [_qcom_read_operand(ctx, _qcom_reg_component_offset_token(src.value, i), lane, src.modifiers) for i in range(4)]
            targets = [f.value for f in inst.fields[1:5]]
        # These move in parallel (swz can swap two registers), so every source is read before any destination is written. Within one
        # kernel loop the scheduler is free to move a load past a store to the same word (two half registers share one), so the sources are
        # parked in a scratch buffer by a first loop and written out by a second one. (_end_lane_range closes the first loop.)
        parked = _end_lane_range(UOp.sink(*[ctx.tmp.index(lane * _c(4, dtypes.int) + _c(i, dtypes.int)).store(_as_reg_bits(v)) for i, v in enumerate(vals)]))
        ctx.advance(parked)
        lane = ctx.range()
        stores: list[UOp] = []
        for i, dst in enumerate(targets):  # chained: half destinations share words, so the read-modify-write stores must not overlap
            stores.append(_write_qcom_store(ctx, dst, lane, ctx.tmp.index(lane * _c(4, dtypes.int) + _c(i, dtypes.int)).load(), after=stores[-1] if stores else None))
        return UOp.sink(*stores)

    if len(inst.fields) < 3: raise RuntimeError(f"Category 1 instruction is missing operands: {inst!r}")
    dst_field, src_field = inst.fields[1], inst.fields[2]
    if op == "cov":
        if "." not in name: raise RuntimeError(f"Conversion instruction lacks type pair: {name!r}")
        pair = _qcom_split_type_pair(name.split(".", 1)[1])
        if pair is None: raise NotImplementedError(f"Unsupported conversion type pair in {name!r}")
        src_type, dst_type = pair
        src = _qcom_u8_source_extends(_qcom_read_typed_operand(ctx, src_field, lane, src_type), src_type, dst_type)
        result = _qcom_apply_instruction_flags(inst, _qcom_encode_typed(src, dst_type))
        return _write_qcom_value(ctx, dst_field.value, lane, result)

    if op == "mov":
        # The mnemonic carries source and destination types (e.g. mov.f32f32 or
        # mov.u32u32). Decode modifiers in the source type, then encode the destination.
        suffix = name.split(".", 1)[1] if "." in name else ""
        pair = _qcom_split_type_pair(suffix)
        if pair is not None:
            src_type, dst_type = pair
            src = _qcom_u8_source_extends(_qcom_read_typed_operand(ctx, src_field, lane, src_type), src_type, dst_type)
            result = _qcom_apply_instruction_flags(inst, _qcom_encode_typed(src, dst_type))
            return _write_qcom_value(ctx, dst_field.value, lane, result)
        result = _qcom_read_operand(ctx, src_field.value, lane, src_field.modifiers)
        return _write_qcom_value(ctx, dst_field.value, lane, _qcom_apply_instruction_flags(inst, result))
    if op in {"mova", "mova1", "mova.r"}:
        src = _qcom_read_operand(ctx, src_field.value, lane, src_field.modifiers)
        # MOVA variants use the same scalar value path; their address-register target
        # is handled by _write_qcom_store.
        return _write_qcom_value(ctx, dst_field.value, lane, src)
    if op == "movs":
        raise NotImplementedError("movs invocation semantics are not yet modeled")
    if op == "movmsk":
        raise NotImplementedError("movmsk requires exact predicate-mask destination semantics")
    raise RuntimeError(f"Unsupported category 1 instruction: {name!r}")


_BIT_DIGITS = bytes.maketrans(b'\x00\x01', b'01')
_QCOM_REG_RE = re.compile(r"(?P<kind>[a-zA-Z]+)(?P<reg>\d+)(?:\.(?P<dim>x|y|z|w))?", re.IGNORECASE)


@functools.cache
def _qcom_reg_token(token: str) -> tuple[str, int, int]:
    token, _ = _split_qcom_modifiers(token)
    match = _QCOM_REG_RE.fullmatch(token.strip())
    if match is None: raise ValueError(f"Unsupported QCOM register token: {token!r}")
    kind = match.group("kind").lower()
    reg = int(match.group("reg"))
    dim = {"x": 0, "y": 1, "z": 2, "w": 3}.get(match.group("dim") or "x", 0)
    # a6xx has 48 per-lane GPRs; the registers numbered 48-55 are the wave-shared file (ir3 prints them as r48.x..r55.w)
    if kind in {"r", "hr"} and reg >= 48: return "s" if kind == "r" else "hs", reg - 48, dim
    return kind, reg, dim


def _qcom_read_value(ctx: _Ctx, token: str, lane: UOp) -> UOp:
    kind, reg, dim = _qcom_reg_token(token)
    if kind == "r": return ctx.read_r_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane)
    if kind == "hr": return ctx.read_r_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane, half=True)
    if kind in {"s", "hs"}: return ctx.read_s_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), half=(kind == "hs"))
    if kind == "c": return ctx.read_c_reg(_c(reg, dtypes.int), _c(dim, dtypes.int))
    # CONSTANT_DEMOTION_ENABLE (set by ops_qcom for IR3): hcN.c is the full 32-bit constant cN.c, demoted for the half-precision reader.
    # Integer readers see the low 16 bits (u2u16, done here); float readers convert f32->f16 (_qcom_demoted_const_f16). ir3_cp.c.
    if kind == "hc": return ctx.read_c_reg(_c(reg, dtypes.int), _c(dim, dtypes.int)) & _c(0xFFFF, dtypes.uint32)
    if kind == "a": return ctx.read_a_reg(_c(reg, dtypes.int))
    if kind == "p": return ctx.read_p_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane)
    raise ValueError(f"Unsupported QCOM register kind {kind!r} in {token!r}")


def _parse_qcom_imm(token: str) -> UOp:
    token = token.strip()
    if token.lower().startswith("h("): token = token[1:]  # half-precision immediate, h(255) / h(1.5) (ir3-common.xml #mulitsrc-immed, cat1 #cat1-immed-src)
    if token.startswith("(") and token.endswith(")"): token = token[1:-1]
    if token.lower() in _QCOM_FLUT_IMMS: return UOp.const(_QCOM_FLUT_IMMS[token.lower()], dtypes.float32)
    if token in {"0", "1"}:
        return _c(int(token), dtypes.uint32)
    if token.lower().lstrip("+-") in {"inf", "nan"}: return UOp.const(float(token), dtypes.float32)
    try:
        if token.lower().lstrip("+-").startswith("0x"):
            return _c(int(token, 0) & 0xFFFFFFFF, dtypes.uint32)
        if token.lower().endswith("f") or "." in token or "e" in token.lower():
            return UOp.const(float(token.rstrip("fF")), dtypes.float32)
        return _c(int(token, 10), dtypes.int32)
    except ValueError:
        raise ValueError(f"Unsupported immediate literal {token!r}")


def _qcom_read_raw_operand(ctx: _Ctx, token: str, lane: UOp, modifiers: list[str] | tuple[str, ...] | None = None) -> tuple[UOp, list[str]]:
    token, token_mods = _split_qcom_modifiers(token)
    mods = list(token_mods) + list(modifiers or [])
    try:
        return _qcom_read_value(ctx, token, lane), mods
    except ValueError:
        # Only non-register tokens are immediates. Do not hide malformed/unsupported
        # register classes by falling through to the numeric literal parser.
        if _QCOM_REG_RE.fullmatch(token.strip()) is not None or "[" in token or "]" in token:
            raise
        return _parse_qcom_imm(token), mods


def _qcom_read_operand(ctx: _Ctx, token: str, lane: UOp, modifiers: list[str] | tuple[str, ...] | None = None) -> UOp:
    value, mods = _qcom_read_raw_operand(ctx, token, lane, modifiers)
    return _apply_qcom_operand_modifiers(value, mods)


def _qcom_read_typed_operand(ctx: _Ctx, field, lane: UOp, typ: str) -> UOp:
    raw, mods = _qcom_read_raw_operand(ctx, field.value, lane, field.modifiers)
    if typ == "f16" and _QCOM_REG_RE.fullmatch(field.value.strip()) and _qcom_reg_token(field.value)[0] == "hc":
        return _apply_qcom_operand_modifiers(_qcom_demoted_const_f16(ctx, field.value), mods)
    # signed integer ALU ops on half registers / h(imm) sources work on 16-bit values: sign-extend from bit 15, not from bit 31
    if typ == "s32" and field.value.strip().lower().startswith("h"): return _apply_qcom_operand_modifiers(_qcom_decode_typed(raw, "s16").cast(dtypes.int32), mods)
    return _apply_qcom_operand_modifiers(_qcom_decode_typed(raw, typ), mods)

def _round_nearest_even(x: UOp) -> UOp:
    # A6XX rndne.f matches IEEE-like round-to-nearest-even: ties go to the even integer.
    truncated = UOp(Ops.TRUNC, src=(x,))
    frac = x - truncated
    abs_frac = frac.abs()
    half = UOp.const(0.5, x.dtype)
    is_half = abs_frac.eq(half)
    is_odd = (truncated.cast(dtypes.int32) & _c(1, dtypes.int32)).ne(_c(0, dtypes.int32))
    # For ties, choose the even value; otherwise round away from zero when |frac| > 0.5.
    nearest = (abs_frac > half).where((x < UOp.const(0.0, x.dtype)).where(truncated - _c(1.0, x.dtype), truncated + _c(1.0, x.dtype)), truncated)
    return is_half.where((is_odd).where((x < UOp.const(0.0, x.dtype)).where(truncated - _c(1.0, x.dtype), truncated + _c(1.0, x.dtype)), truncated), nearest)


def _round_away_from_zero(x: UOp) -> UOp:
    # A6XX rndaz.f rounds to the nearest integer away from zero.
    sign = (x < UOp.const(0.0, x.dtype)).where(UOp.const(-1.0, x.dtype), UOp.const(1.0, x.dtype))
    return UOp(Ops.TRUNC, src=(x + (UOp.const(0.5, x.dtype) * sign),)).cast(x.dtype)


def _qcom_cmp(lhs: UOp, rhs: UOp, condition: str) -> UOp:
    ops = {"eq": lhs.eq(rhs), "ne": lhs.ne(rhs), "lt": lhs < rhs, "le": lhs <= rhs, "gt": lhs > rhs, "ge": lhs >= rhs}
    if condition not in ops: raise RuntimeError(f"Unsupported comparison condition {condition!r}")
    return ops[condition]


def _qcom_clz32(value: UOp) -> UOp:
    value = value.cast(dtypes.uint32)
    count = _c(0, dtypes.uint32)
    for bit in range(32): count = count + (value >> _c(bit, dtypes.uint32)).eq(_c(0, dtypes.uint32)).cast(dtypes.uint32)
    return count


def _qcom_popcount32(value: UOp) -> UOp:
    value = value.cast(dtypes.uint32)
    value = value - ((value >> _c(1, dtypes.uint32)) & _c(0x55555555, dtypes.uint32))
    value = (value & _c(0x33333333, dtypes.uint32)) + ((value >> _c(2, dtypes.uint32)) & _c(0x33333333, dtypes.uint32))
    value = (value + (value >> _c(4, dtypes.uint32))) & _c(0x0F0F0F0F, dtypes.uint32)
    value = value + (value >> _c(8, dtypes.uint32))
    value = value + (value >> _c(16, dtypes.uint32))
    return value & _c(0x3F, dtypes.uint32)


def _qcom_bitreverse32(value: UOp) -> UOp:
    value = value.cast(dtypes.uint32)
    out = _c(0, dtypes.uint32)
    for bit in range(32): out = out | (((value >> _c(bit, dtypes.uint32)) & _c(1, dtypes.uint32)) << _c(31-bit, dtypes.uint32))
    return out


def _qcom_sign_extend(value: UOp, bits: int) -> UOp:
    shift = 32 - bits
    return (value.cast(dtypes.int32) << _c(shift, dtypes.int32)) >> _c(shift, dtypes.int32)


def _compile_cat2(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 2 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    if name == "nop": return UOp.sink()
    lane = ctx.range()
    if name == "setrm":
        # The rounding-mode register is not modelled; silently ignoring it would make
        # round-sensitive shaders misleading, so keep it explicitly unsupported.
        raise NotImplementedError("setrm changes rounding state, which is not yet modeled")
    if name.startswith(("bary.", "mgen.", "shb", "msad")):
        raise NotImplementedError(f"Category 2 instruction {name!r} requires special execution semantics")
    if name in {"flat.b", "mull.u2", "div2.u", "mul2.u"}:
        raise NotImplementedError(f"Category 2 instruction {name!r} is not implemented")
    if name not in {"nop"} and len(inst.fields) < 3: raise RuntimeError(f"Category 2 instruction is missing operands: {inst!r}")
    dst_token = inst.fields[1].value  # len(inst.fields) >= 3 is guaranteed above
    src_fields = inst.fields[2:]
    src_vals = [] if name == "absneg.s" else [_qcom_read_operand(ctx, f.value, lane, f.modifiers) for f in src_fields]  # absneg.s decodes as signed below

    unary = {"absneg.f", "absneg.s", "sign.f", "floor.f", "ceil.f", "rndne.f", "rndaz.f", "trunc.f", "not.b", "bfrev.b", "clz.s", "clz.b", "cbits.b"}
    if name in unary:
        if len(src_fields) != 1: raise RuntimeError(f"{name} expects one source: {inst!r}")
        if name.endswith(".f"):
            srcf = _qcom_float_operand(ctx, src_fields[0], lane)
        elif name.endswith(".s"):
            srcf = _qcom_read_typed_operand(ctx, src_fields[0], lane, "s32")
        else:
            srcf = src_vals[0]
        # absneg is the unary ALU route that consumes source abs/neg modifiers;
        # the opcode itself does not unconditionally apply abs().
        if name == "absneg.f": result = srcf
        elif name == "absneg.s": result = srcf  # srcf is the signed-int view with (abs)/(neg) already applied

        elif name == "sign.f": result = (srcf < UOp.const(0.0, dtypes.float32)).where(UOp.const(-1.0, dtypes.float32), (srcf > UOp.const(0.0, dtypes.float32)).where(UOp.const(1.0, dtypes.float32), UOp.const(0.0, dtypes.float32)))
        elif name == "floor.f": result = srcf.floor()
        elif name == "ceil.f": result = srcf.ceil()
        elif name == "rndne.f": result = _round_nearest_even(srcf)
        elif name == "rndaz.f": result = _round_away_from_zero(srcf)
        elif name == "trunc.f": result = srcf.trunc()
        elif name == "not.b": result = ~src_vals[0].cast(dtypes.uint32)
        elif name == "bfrev.b": result = _qcom_bitreverse32(src_vals[0])
        elif name == "clz.b":
            # Mesa ISA doc: "Number of leading zeroes, or -1 if the input was 0." (nir ufind_msb_rev / find_lsb rely on the -1)
            v = src_vals[0].cast(dtypes.uint32)
            result = v.eq(_c(0)).where(_c(0xFFFFFFFF), _qcom_clz32(v))
        elif name == "clz.s":
            # Mesa ISA doc: "Number of leading bits (including sign) that match the sign bit, or -1 for input of 0/~0."
            v = src_vals[0].cast(dtypes.int32)
            flipped = (v ^ (v >> _c(31, dtypes.int32))).cast(dtypes.uint32)  # leading sign bits become leading zeros
            result = flipped.eq(_c(0)).where(_c(0xFFFFFFFF), _qcom_clz32(flipped))
        elif name == "cbits.b": result = _qcom_popcount32(src_vals[0])
        else: raise RuntimeError(f"Unhandled cat2 unary op {name!r}")
        return _qcom_write_result(ctx, dst_token, lane, _qcom_apply_instruction_flags(inst, result))

    if name.startswith(("cmpv.", "cmps.")):
        if len(src_fields) != 2: raise RuntimeError(f"{name} expects two source operands: {inst!r}")
        parts = name.split(".")
        if len(parts) != 3: raise NotImplementedError(f"Unsupported comparison form {name!r}")
        cmp_kind, cmp_type, condition = parts
        if cmp_type == "f": lhs, rhs = _qcom_float_operand(ctx, src_fields[0], lane), _qcom_float_operand(ctx, src_fields[1], lane)
        elif cmp_type == "u":
            lhs = _qcom_read_typed_operand(ctx, src_fields[0], lane, "u32")
            rhs = _qcom_read_typed_operand(ctx, src_fields[1], lane, "u32")
        elif cmp_type == "s":
            lhs = _qcom_read_typed_operand(ctx, src_fields[0], lane, "s32")
            rhs = _qcom_read_typed_operand(ctx, src_fields[1], lane, "s32")
        else: raise RuntimeError(f"Unknown comparison type in {name!r}")
        pred = _qcom_cmp(lhs, rhs, condition)
        # cmpv is the vector-mask form (0xffffffff/0); cmps returns a 0/1 value.
        true_value = 0xFFFFFFFF if cmp_kind == "cmpv" else 1
        result = pred.where(_c(true_value, dtypes.uint32), _c(0, dtypes.uint32))
        return _qcom_write_result(ctx, dst_token, lane, _qcom_apply_instruction_flags(inst, result))

    if len(src_fields) != 2: raise RuntimeError(f"{name} expects two source operands: {inst!r}")
    a, b = src_vals
    float_a = _qcom_float_operand(ctx, src_fields[0], lane)
    float_b = _qcom_float_operand(ctx, src_fields[1], lane)
    signed_a = _qcom_read_typed_operand(ctx, src_fields[0], lane, "s32")
    signed_b = _qcom_read_typed_operand(ctx, src_fields[1], lane, "s32")
    if name == "add.f": result = float_a + float_b
    elif name in {"add.u", "add.s"}: result = (a.cast(dtypes.uint32) + b.cast(dtypes.uint32)) if name.endswith(".u") else (signed_a + signed_b)
    elif name in {"sub.u", "sub.s"}: result = (a.cast(dtypes.uint32) - b.cast(dtypes.uint32)) if name.endswith(".u") else (signed_a - signed_b)
    elif name == "mul.f": result = float_a * float_b
    elif name == "min.f": result = float_a.minimum(float_b)
    elif name == "max.f": result = float_a.maximum(float_b)
    elif name in {"min.u", "max.u", "min.s", "max.s"}:
        dtype = dtypes.uint32 if name.endswith(".u") else dtypes.int32
        lhs, rhs = (a.cast(dtype), b.cast(dtype)) if dtype == dtypes.uint32 else (signed_a, signed_b)
        result = lhs.minimum(rhs) if name.startswith("min") else lhs.maximum(rhs)
    elif name in {"mul.u24", "mul.s24"}:
        if name == "mul.u24": result = (a.cast(dtypes.uint32) & _c(0xFFFFFF, dtypes.uint32)) * (b.cast(dtypes.uint32) & _c(0xFFFFFF, dtypes.uint32))
        else: result = _qcom_sign_extend(signed_a, 24) * _qcom_sign_extend(signed_b, 24)
    elif name == "and.b": result = a.cast(dtypes.uint32) & b.cast(dtypes.uint32)
    elif name == "or.b": result = a.cast(dtypes.uint32) | b.cast(dtypes.uint32)
    elif name == "xor.b": result = a.cast(dtypes.uint32) ^ b.cast(dtypes.uint32)
    elif name == "shl.b": result = a.cast(dtypes.uint32) << (b.cast(dtypes.uint32) & _c(31, dtypes.uint32))
    elif name == "shr.b": result = a.cast(dtypes.uint32) >> (b.cast(dtypes.uint32) & _c(31, dtypes.uint32))
    elif name == "ashr.b": result = signed_a >> (b.cast(dtypes.uint32) & _c(31, dtypes.uint32))  # signed_a: s16-extended for half sources
    elif name == "getbit.b": result = (a.cast(dtypes.uint32) >> (b.cast(dtypes.uint32) & _c(31, dtypes.uint32))) & _c(1, dtypes.uint32)
    elif name == "mull.u":
        # Mesa lowers nir umul_16x16 to mull.u: unsigned product of the low 16 bits of each source (32-bit result)
        result = (a.cast(dtypes.uint32) & _c(0xFFFF)) * (b.cast(dtypes.uint32) & _c(0xFFFF))
    else: raise RuntimeError(f"Unsupported category 2 instruction: {name!r}")
    return _qcom_write_result(ctx, dst_token, lane, _qcom_apply_instruction_flags(inst, result))


def _qcom_mem_dtype(suffix: str) -> Any:
    suffix = suffix.strip().lower()
    if suffix in {"", "a", "k"}: return dtypes.uint32
    mapping = {
        "u8": dtypes.uint8, "u8_32": dtypes.uint8, "s8": dtypes.int8,
        "u16": dtypes.uint16, "s16": dtypes.int16,
        "u32": dtypes.uint32, "s32": dtypes.int32,
        "f16": dtypes.float16, "f32": dtypes.float32,
        "u64": dtypes.uint64, "s64": dtypes.int64,
    }
    if suffix not in mapping:
        raise ValueError(f"Unsupported QCOM memory type suffix: {suffix!r}")
    return mapping[suffix]


def _mem_byte_index(ctx: _Ctx, mem_kind: str, addr_i: UOp, i: int, lane: UOp) -> UOp:
    """Byte index of byte i of an access inside the memory buffer. Private memory is per lane: the emulator keeps one buffer per
    wave and interleaves it by dword across the lanes (lane l's dword d lives at d*wave_size*4 + l*4), so every lane sees its own
    p[...] space. The GPU VA in SP_CS_PVT_MEM_BASE is never read by shaders, so it does not need to be honoured."""
    b = addr_i + _c(i, dtypes.int64)
    if mem_kind != "p": return b
    return (b >> _c(2, dtypes.int64)) * _c(ctx.wave_size * 4, dtypes.int64) + lane.cast(dtypes.int64) * _c(4, dtypes.int64) + (b & _c(3, dtypes.int64))

def _load_mem_value(ctx: _Ctx, mem_kind: str, addr: UOp, suffix: str, lane: UOp) -> UOp:
    mem = {"g": ctx.vmem, "l": ctx.shared, "p": ctx.pvt_mem}[mem_kind]
    active = _thread_active(ctx.read_exec_mask(), lane)
    dtype = _qcom_mem_dtype(suffix)
    nbytes = dtype.itemsize
    raw_dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}[nbytes]
    addr_i = addr.cast(dtypes.int64)
    safe_addr = active.where(addr_i, _c(0, dtypes.int64))
    raw: UOp | None = None
    for i in range(nbytes):
        byte = mem.index(_mem_byte_index(ctx, mem_kind, safe_addr, i, lane).valid(active)).load().cast(raw_dtype)
        part = byte if i == 0 else byte << _c(8 * i, raw_dtype)
        raw = part if raw is None else raw | part
    assert raw is not None  # nbytes >= 1
    # Memory contains bits. Reinterpret signed and floating-point values rather than numerically casting them.
    if dtype in {dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.float16, dtypes.float32}:
        decoded = raw.bitcast(dtype)
        # Narrow signed loads are sign-extended into the 32-bit register payload.
        if dtype in {dtypes.int8, dtypes.int16}: return decoded.cast(dtypes.int32)
        return decoded
    return raw.cast(dtype)


def _store_mem_value(ctx: _Ctx, mem_kind: str, addr: UOp, value: UOp, suffix: str, lane: UOp):
    mem = {"g": ctx.vmem, "l": ctx.shared, "p": ctx.pvt_mem}[mem_kind]
    active = _thread_active(ctx.read_exec_mask(), lane)
    dtype = _qcom_mem_dtype(suffix)
    nbytes = dtype.itemsize
    raw_dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}[nbytes]
    if value.dtype == raw_dtype:
        raw = value
    elif value.dtype in {dtypes.float16, dtypes.float32} and dtype in {dtypes.float16, dtypes.float32}:
        typed = value.cast(dtype)
        raw = typed.bitcast(raw_dtype)
    elif value.dtype.itemsize == nbytes:
        raw = value.bitcast(raw_dtype)
    else:
        raw = value.cast(raw_dtype)
    addr_i = addr.cast(dtypes.int64)
    stores = []
    for i in range(nbytes):
        byte = (raw >> _c(8 * i, raw_dtype)).cast(dtypes.uint8) if i else raw.cast(dtypes.uint8)
        stores.append(mem.index(_mem_byte_index(ctx, mem_kind, addr_i, i, lane).valid(active)).store(byte))
    return stores


def _compile_cat3(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 3 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    if len(inst.fields) != 5: raise RuntimeError(f"Category 3 instruction requires destination and three sources: {inst!r}")
    if name in {"madsh.u16", "mad.f16.mul2", "mad.f32.mul2", "mad.f16.div2", "mad.f32.div2"}:
        raise NotImplementedError(f"Category 3 instruction {name!r} requires ISA-specific packed/scale semantics")
    lane = ctx.range()
    dst = inst.fields[1].value
    src_fields = inst.fields[2:5]
    vals = [_qcom_read_operand(ctx, f.value, lane, f.modifiers) for f in src_fields]
    a, b, c = vals
    parts = name.split(".")

    def f16(field, value): return _qcom_read_typed_operand(ctx, field, lane, "f16").cast(dtypes.float32)
    def s24(value): return _qcom_sign_extend(value, 24)
    def packed_byte(value, byte, signed=True):
        v = (value.cast(dtypes.uint32) >> _c(byte * 8, dtypes.uint32)) & _c(0xFF, dtypes.uint32)
        return v.cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.int32) if signed else v

    if name in {"mad.f16", "mad.f32"}:
        if name.endswith("f16"):
            value = (f16(src_fields[0], a) * f16(src_fields[1], b) + f16(src_fields[2], c)).cast(dtypes.float16)
        else: value = _qcom_float_operand(ctx, src_fields[0], lane) * _qcom_float_operand(ctx, src_fields[1], lane) + _qcom_float_operand(ctx, src_fields[2], lane)
    elif name == "madsh.m16":
        # Mesa nir_op_imadsh_mix16 (ir3_compiler_nir.c shared-reg fallback): ((a & 0xffff) * (b >> 16)) << 16, plus c. imul is
        # mull.u t,a,b; madsh.m16 t,a,b,t; madsh.m16 d,b,a,t.
        a32, b32 = a.cast(dtypes.uint32), b.cast(dtypes.uint32)
        value = (((a32 & _c(0xFFFF, dtypes.uint32)) * (b32 >> _c(16, dtypes.uint32))) << _c(16, dtypes.uint32)) + c.cast(dtypes.uint32)
    elif name == "mad.u16": value = (a.cast(dtypes.uint16).cast(dtypes.uint32) * b.cast(dtypes.uint16).cast(dtypes.uint32) + c.cast(dtypes.uint32)).cast(dtypes.uint32)
    elif name == "mad.s16": value = (a.cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32) * b.cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32) + c.cast(dtypes.int32)).cast(dtypes.int32)
    elif name == "mad.u24": value = ((a.cast(dtypes.uint32) & _c(0xFFFFFF, dtypes.uint32)) * (b.cast(dtypes.uint32) & _c(0xFFFFFF, dtypes.uint32)) + c.cast(dtypes.uint32)).cast(dtypes.uint32)
    elif name == "mad.s24": value = (s24(a) * s24(b) + c.cast(dtypes.int32)).cast(dtypes.int32)
    elif name in {"sel.b16", "sel.b32"}:
        # Mesa emits ir3_SEL_B32(b, <value if true>, <cond>, <value if false>) for nir icsel (ir3_compiler_nir.c): the middle
        # source is the condition (non-zero selects src1), not the first.
        value = b.cast(dtypes.uint32).ne(_c(0)).where(a.cast(dtypes.uint32), c.cast(dtypes.uint32))
    elif name in {"sel.s16", "sel.s32", "sel.f16", "sel.f32"}:
        raise NotImplementedError(f"{name}: operand roles and condition are not documented in Mesa's ISA XML and Mesa never emits them")
    elif name in {"sad.s16", "sad.s32"}:
        if name.endswith("s16"):
            aa = a.cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32)
            bb = b.cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32)
        else: aa, bb = a.cast(dtypes.int32), b.cast(dtypes.int32)
        value = (aa - bb).abs() + c.cast(dtypes.int32)
    elif name in {"shrm", "shlm", "shrg", "shlg", "andg"}:
        shift = a.cast(dtypes.uint32) & _c(31, dtypes.uint32)
        if name == "shrm": value = (b.cast(dtypes.uint32) >> shift) & c.cast(dtypes.uint32)
        elif name == "shlm": value = (b.cast(dtypes.uint32) << shift) & c.cast(dtypes.uint32)
        elif name == "shrg": value = (b.cast(dtypes.uint32) >> shift) | c.cast(dtypes.uint32)
        elif name == "shlg": value = (b.cast(dtypes.uint32) << shift) | c.cast(dtypes.uint32)
        else: value = (b.cast(dtypes.uint32) & a.cast(dtypes.uint32)) | c.cast(dtypes.uint32)
    elif parts[0] == "dp2acc":
        # Mesa's Cat3 ISA description: SRC1 is i8vec2/u8vec2, SRC2 is u8vec2;
        # the low/high suffix selects which byte pair is consumed from each word.
        signed = "mixed" in parts[1:]
        pack = "high" if "high" in parts[1:] else "low"
        first_byte = 2 if pack == "high" else 0
        a0, a1 = packed_byte(a, first_byte, signed), packed_byte(a, first_byte + 1, signed)
        b0, b1 = packed_byte(b, first_byte, False), packed_byte(b, first_byte + 1, False)
        value = c.cast(dtypes.int32) + a0*b0 + a1*b1
    elif parts[0] == "dp4acc":
        # DP4 consumes all four bytes in each 32-bit operand. The low form is the
        # A630 form emitted by Mesa; reject a high-form rather than guessing its pair layout.
        if "high" in parts[1:]:
            raise NotImplementedError("dp4acc.high needs the 64-bit packed-source register model")
        signed = "mixed" in parts[1:]
        value = c.cast(dtypes.int32)
        for i in range(4): value = value + packed_byte(a, i, signed) * packed_byte(b, i, False)
    else: raise RuntimeError(f"Unsupported category 3 instruction: {name!r}")

    return _qcom_write_result(ctx, dst, lane, _qcom_apply_instruction_flags(inst, value))


def _compile_cat4(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 4 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    if len(inst.fields) != 3: raise RuntimeError(f"Category 4 instruction expects destination and source: {inst!r}")
    lane = ctx.range()
    dst, src_field = inst.fields[1].value, inst.fields[2]
    half_op = name in {"hrsq", "hlog2", "hexp2"}
    x = _qcom_read_typed_operand(ctx, src_field, lane, "f16").cast(dtypes.float32) if half_op else _qcom_float_operand(ctx, src_field, lane)
    if name in {"rcp", "hrsq", "rsq"}: result = x.sqrt().reciprocal() if name in {"rsq", "hrsq"} else x.reciprocal()
    elif name in {"log2", "hlog2"}: result = x.log2()
    elif name in {"exp2", "hexp2"}: result = x.exp2()
    elif name == "sin": result = x.sin()
    elif name == "cos": result = x.cos()
    elif name == "sqrt": result = x.sqrt()
    else: raise RuntimeError(f"Unsupported category 4 instruction: {name!r}")
    if name.startswith("h"): result = result.cast(dtypes.float16)
    return _qcom_write_result(ctx, dst, lane, _qcom_apply_instruction_flags(inst, result))


def _image_descriptor(ctx, buf: UOp, index: int) -> dict:
    """Decode the A6XX_TEX_CONST descriptor `index` of a texture (SB6_CS_TEX) or IBO (SB6_CS_SHADER/ST6_UAV) array at run time: the
    descriptors change with every dispatch, so they are read from a buffer rather than baked into the (cached) kernel."""
    w = [buf.index(_c(index * 16 + k, dtypes.int)).load() for k in range(6)]
    def field(word, mask, shift): return (w[word] & _c(mask, dtypes.uint32)) >> _c(shift, dtypes.uint32)
    base = (w[4] & _c(mesa.A6XX_TEX_CONST_4_BASE_LO__MASK, dtypes.uint32)).cast(dtypes.uint64) | ((w[5] & _c(mesa.A6XX_TEX_CONST_5_BASE_HI__MASK, dtypes.uint32)).cast(dtypes.uint64) << _c(32, dtypes.uint64))
    return {"w0": w[0], "base": base, "fmt": field(0, mesa.A6XX_TEX_CONST_0_FMT__MASK, mesa.A6XX_TEX_CONST_0_FMT__SHIFT),
            "width": field(1, mesa.A6XX_TEX_CONST_1_WIDTH__MASK, mesa.A6XX_TEX_CONST_1_WIDTH__SHIFT).cast(dtypes.int),
            "height": field(1, mesa.A6XX_TEX_CONST_1_HEIGHT__MASK, mesa.A6XX_TEX_CONST_1_HEIGHT__SHIFT).cast(dtypes.int),
            "pitch": field(2, mesa.A6XX_TEX_CONST_2_PITCH__MASK, mesa.A6XX_TEX_CONST_2_PITCH__SHIFT).cast(dtypes.uint64)}


def _texel_addr(desc: dict, x: UOp, y: UOp) -> tuple[UOp, UOp, UOp]:
    """(byte address of texel (x, y), in-bounds, is_32bit_float_format) for a linear 2D image of RGBA32F or RGBA16F texels."""
    inb = (x >= _c(0, dtypes.int)) & (x < desc["width"]) & (y >= _c(0, dtypes.int)) & (y < desc["height"])
    is32 = desc["fmt"].eq(_c(mesa.FMT6_32_32_32_32_FLOAT, dtypes.uint32))
    bpp = is32.where(_c(16, dtypes.uint64), _c(8, dtypes.uint64))
    return desc["base"] + y.cast(dtypes.uint64) * desc["pitch"] + x.cast(dtypes.uint64) * bpp, inb, is32


def _load_texel(ctx, desc: dict, x: UOp, y: UOp, lane: UOp, swizzle: bool) -> list[UOp]:
    """The four float32 components of texel (x, y); zero outside the image (CLAMP_TO_BORDER with a transparent-black border)."""
    addr, inb, is32 = _texel_addr(desc, x, y)
    valid = inb & _thread_active(ctx.read_exec_mask(), lane)
    comps = []
    for i in range(4):
        raw: UOp | None = None
        for j in range(4):  # 4 bytes of an f32 component, only the first 2 of an f16 one (never read past the texel)
            off = is32.where(_c(4 * i + j, dtypes.uint64), _c(2 * i + j, dtypes.uint64))
            byte = ctx.vmem.index((addr + off).cast(dtypes.int64).valid(valid & (is32 | _c(j < 2, dtypes.bool)))).load().cast(dtypes.uint32)
            raw = byte if raw is None else raw | (byte << _c(8 * j, dtypes.uint32))
        assert raw is not None
        comps.append(is32.where(raw.bitcast(dtypes.float32), raw.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)))
    if not swizzle: return comps
    out = []
    for shift, mask in ((mesa.A6XX_TEX_CONST_0_SWIZ_X__SHIFT, mesa.A6XX_TEX_CONST_0_SWIZ_X__MASK), (mesa.A6XX_TEX_CONST_0_SWIZ_Y__SHIFT, mesa.A6XX_TEX_CONST_0_SWIZ_Y__MASK),
                        (mesa.A6XX_TEX_CONST_0_SWIZ_Z__SHIFT, mesa.A6XX_TEX_CONST_0_SWIZ_Z__MASK), (mesa.A6XX_TEX_CONST_0_SWIZ_W__SHIFT, mesa.A6XX_TEX_CONST_0_SWIZ_W__MASK)):
        sel = (desc["w0"] & _c(mask, dtypes.uint32)) >> _c(shift, dtypes.uint32)  # 0..3 = R,G,B,A, 4 = zero, 5 = one
        v = sel.eq(_c(4, dtypes.uint32)).where(UOp.const(0.0, dtypes.float32), UOp.const(1.0, dtypes.float32))
        for c in (3, 2, 1, 0): v = sel.eq(_c(c, dtypes.uint32)).where(comps[c], v)
        out.append(v)
    return out


def _store_texel(ctx, desc: dict, x: UOp, y: UOp, values: list[UOp], lane: UOp) -> list[UOp]:
    """Write four float32 values as texel (x, y); out of bounds writes are dropped."""
    addr, inb, is32 = _texel_addr(desc, x, y)
    valid = inb & _thread_active(ctx.read_exec_mask(), lane)
    stores = []
    for j in range(16):
        f32_byte = (values[j // 4].bitcast(dtypes.uint32) >> _c(8 * (j % 4), dtypes.uint32)) & _c(0xFF, dtypes.uint32)
        f16_comp = j // 2
        f16_byte = ((values[f16_comp].cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32) >> _c(8 * (j % 2), dtypes.uint32)) & _c(0xFF, dtypes.uint32)) if j < 8 else _c(0, dtypes.uint32)
        byte = is32.where(f32_byte, f16_byte).cast(dtypes.uint8)
        stores.append(ctx.vmem.index((addr + _c(j, dtypes.uint64)).cast(dtypes.int64).valid(valid & (is32 | _c(j < 8, dtypes.bool)))).store(byte))
    return stores


def _write_texel_result(ctx, lane: UOp, dst: str, comps: list[UOp], wrmask: str, typ: str) -> UOp:
    """Results of a texture/image load are written to consecutive registers, one per set bit of the write mask."""
    if typ not in {"f32", "f16"}: raise NotImplementedError(f"image/texture data type {typ!r} is not implemented")
    stores: list[UOp] = []
    for n, c in enumerate("xyzw"):
        if c not in wrmask: continue
        v = comps["xyzw".index(c)]
        stores.append(_write_qcom_store(ctx, _qcom_reg_component_offset_token(dst, len(stores)), lane, v.cast(dtypes.float16) if typ == "f16" else v, after=stores[-1] if stores else None))
    return UOp.sink(*stores)


def _compile_cat5(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 5 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    # Only isam (integer-coordinate texel fetch, which is what ir3 emits for a read-only image_load) is modelled. The rest of cat5 (filtered
    # sampling, gathers, derivatives, bindless / indirect s2en forms) needs sampler state or quad execution semantics, so it must not be faked.
    if name != "isam.1d" or len(inst.fields) != 5 or not (inst.fields[3].value.startswith("s#") and inst.fields[4].value.startswith("t#")):
        raise NotImplementedError(f"Category 5 instruction {name!r} requires sampler state or quad/subgroup execution semantics")
    dst = inst.fields[1]
    typ = next((m for m in dst.modifiers if m in _QCOM_TEX_TYPES), "f32")
    wrmask = next((m for m in dst.modifiers if re.fullmatch(r"[xyzw]{1,4}", m)), "xyzw")
    lane = ctx.range()
    x = _qcom_read_operand(ctx, inst.fields[2].value, lane).cast(dtypes.uint32).bitcast(dtypes.int32)
    y = _qcom_read_operand(ctx, _qcom_reg_component_offset_token(inst.fields[2].value, 1), lane).cast(dtypes.uint32).bitcast(dtypes.int32)
    desc = _image_descriptor(ctx, ctx.tex, int(inst.fields[4].value[2:]))
    return _write_texel_result(ctx, lane, dst.value, _load_texel(ctx, desc, x, y, lane, swizzle=True), wrmask, typ)


def _compile_image_buffer_op(inst, ctx):
    """cat6 stib / ldib: typed 2D image store / load through an IBO descriptor, e.g. `stib.b.typed.2d.f32.4.imm r2.y, r1.w, 0`."""
    name = _inst_name(inst).strip().lower()
    parts = name.split(".")
    if len(parts) != 7 or parts[1:4] != ["b", "typed", "2d"] or parts[4] not in {"f32", "f16"} or parts[5] != "4" or parts[6] != "imm" or len(inst.fields) != 4:
        raise NotImplementedError(f"image instruction {name!r} (only typed 2d f32/f16 x4 with an immediate IBO index is implemented)")
    op, typ = parts[0], parts[4]
    lane = ctx.range()
    reg, coord = inst.fields[1].value, inst.fields[2].value
    if op == "stib" and typ == "f16" and _qcom_reg_token(reg)[0] == "r": reg = "h" + reg  # the disassembler omits the "h" on store sources
    x = _qcom_read_operand(ctx, coord, lane).cast(dtypes.uint32).bitcast(dtypes.int32)
    y = _qcom_read_operand(ctx, _qcom_reg_component_offset_token(coord, 1), lane).cast(dtypes.uint32).bitcast(dtypes.int32)
    desc = _image_descriptor(ctx, ctx.ibo, int(inst.fields[3].value, 0))
    if op == "ldib": return _write_texel_result(ctx, lane, reg, _load_texel(ctx, desc, x, y, lane, swizzle=False), "xyzw", typ)
    reads = [_qcom_read_operand(ctx, _qcom_reg_component_offset_token(reg, i), lane) for i in range(4)]
    values = [(_as_float16(r) if typ == "f16" else _as_float32(r)) for r in reads]
    return UOp.sink(*_store_texel(ctx, desc, x, y, values, lane))


def _qcom_read_mem_address(ctx, token: str, lane: UOp) -> tuple[str, UOp]:
    """Decode a g[...]/l[...]/p[...] operand as printed by the Mesa ir3 disassembler.
    Global (a6xx, 64-bit GPU VA): the base is a register pair, low word in the named register and high word in the next
    consecutive register (components carry x->y->z->w->next r), plus an immediate byte offset (ir3-cat6.xml: ldg/stg OFF;
    ir3_a6xx.c collects (addr, base_hi) for ldg). Returns a uint64 address.
    Local/private: 32-bit register base plus offset. Returns a uint32 address."""
    token = token.strip()
    match = re.fullmatch(r"(?P<space>[glp])\[(?P<expr>.+)\]", token, re.IGNORECASE)
    if match is None: raise ValueError(f"Unsupported Cat6 address expression: {token!r}")
    space, expr = match.group("space").lower(), match.group("expr").replace(" ", "")
    terms = re.findall(r"([+-]?)([^+-]+)", expr)
    if not terms: raise ValueError(f"Empty Cat6 address expression: {token!r}")
    if space == "g":
        sign, base = terms[0]
        if sign == "-" or _qcom_reg_token(base)[0] != "r":
            raise NotImplementedError(f"global address base must be a full r register pair, got {base!r}")
        lo = _qcom_read_operand(ctx, base, lane).cast(dtypes.uint32).cast(dtypes.uint64)
        hi = _qcom_read_operand(ctx, _qcom_reg_component_offset_token(base, 1), lane).cast(dtypes.uint32).cast(dtypes.uint64)
        addr = lo | (hi << _c(32, dtypes.uint64))
        for sign, term in terms[1:]:
            try: imm = int(term, 0)
            except ValueError as e: raise NotImplementedError(f"global address offset must be an immediate, got {term!r}") from e
            addr = addr + _c((-imm if sign == "-" else imm) & 0xFFFFFFFFFFFFFFFF, dtypes.uint64)  # two's-complement wraparound
        return space, addr
    addr = None
    for sign, term in terms:
        value = _qcom_read_operand(ctx, term, lane).cast(dtypes.uint32)
        if addr is None:
            addr = value if sign != "-" else (_c(0, dtypes.uint32) - value)
        elif sign == "-": addr = addr - value
        else: addr = addr + value
    return space, addr


def _compile_cat6(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 6 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    if name.startswith(("stib", "ldib")): return _compile_image_buffer_op(inst, ctx)
    # .a has a distinct complex-address encoding; .k loads constants through a special path.
    # These are mnemonic components (e.g. ldg.a.f32), not necessarily suffixes.
    mode_parts = name.split(".")[1:]
    if any(part in {"a", "k", "b"} for part in mode_parts):
        raise NotImplementedError(f"Special Cat6 addressing mode {name!r} is not implemented")
    op, _, suffix = name.partition(".")
    mem_kind = {"ldg": "g", "stg": "g", "ldl": "l", "stl": "l", "ldp": "p", "stp": "p"}.get(op)
    if mem_kind is None: raise NotImplementedError(f"Cat6 instruction {name!r} (atomic/resource/other form) is not implemented")
    dtype_suffix = suffix or "u32"
    dtype = _qcom_mem_dtype(dtype_suffix)
    if dtype.itemsize == 8:
        raise NotImplementedError(f"Cat6 {name!r} uses a 64-bit value; paired IR3 register semantics are not implemented")
    # The trailing Cat6 operand is a component count (1..4), not a byte count.
    # We model consecutive vector components in the flattened r/hr register file.
    if len(inst.fields) not in {3, 4}:
        raise RuntimeError(f"{name} expects address/value operands and an optional component count: {inst!r}")
    count = 1
    if len(inst.fields) == 4:
        try: count = int(inst.fields[3].value, 0)
        except ValueError as e: raise NotImplementedError(f"{name} component count must be an immediate: {inst.fields[3].value!r}") from e
    if not 1 <= count <= 4:
        raise NotImplementedError(f"{name} component count {count} is outside the supported range 1..4")

    lane = ctx.range()
    if op.startswith("ld"):
        dst = inst.fields[1].value
        space, addr = _qcom_read_mem_address(ctx, inst.fields[2].value, lane)
        if space != mem_kind: raise RuntimeError(f"{name} uses {mem_kind}[...] but operand addresses {space}[...]")
        if count > 1 and _qcom_reg_token(dst)[0] not in {"r", "hr"}:
            raise NotImplementedError(f"Multi-component {name} destination must be an r/hr register, got {dst!r}")
        stores: list[UOp] = []
        for i in range(count):
            dst_i = _qcom_reg_component_offset_token(dst, i) if count > 1 else dst
            addr_i = addr + _c(i * dtype.itemsize, addr.dtype)
            value = _load_mem_value(ctx, mem_kind, addr_i, dtype_suffix, lane)
            # two half registers share one 32-bit word, so the writes must be ordered (each is a read-modify-write of that word)
            stores.append(_write_qcom_store(ctx, dst_i, lane, value, after=stores[-1] if stores else None))
        return UOp.sink(*stores)

    if op.startswith("st"):
        space, addr = _qcom_read_mem_address(ctx, inst.fields[1].value, lane)
        if space != mem_kind: raise RuntimeError(f"{name} uses {mem_kind}[...] but operand addresses {space}[...]")
        src_token = inst.fields[2].value
        # u8/u16/s16/f16 stores read a half register (ir3-common.xml #type-half), but the disassembler leaves the "h" off stl/stp
        # sources (it is printed for stg), e.g. "stp.u16 p[r24.x+6], r37.y" after "ldg.u16 hr36.z, ..., 4": same register number.
        if dtype_suffix.strip().lower() in {"u8", "u16", "s16", "f16"} and _qcom_reg_token(src_token)[0] == "r": src_token = "h" + src_token
        if count > 1 and _qcom_reg_token(src_token)[0] not in {"r", "hr"}:
            raise NotImplementedError(f"Multi-component {name} source must be an r/hr register, got {src_token!r}")
        stores = []
        for i in range(count):
            src_i = _qcom_reg_component_offset_token(src_token, i) if count > 1 else src_token
            src = _qcom_read_operand(ctx, src_i, lane, inst.fields[2].modifiers)
            addr_i = addr + _c(i * dtype.itemsize, addr.dtype)
            stores.extend(_store_mem_value(ctx, mem_kind, addr_i, src, dtype_suffix, lane))
        return UOp.sink(*stores)
    raise RuntimeError(f"Unsupported category 6 instruction: {name!r}")


def _compile_cat7(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 7 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    op = name.split(".", 1)[0]
    if op == "bar": return UOp.sink()  # wave scheduler performs the workgroup rendezvous
    # A6xx disassembly includes qualifiers such as bar.g, fence.g.l.r.w, dccln.all,
    # and sleep.l. CPU-backed buffers are coherent and runners are synchronous, so these
    # cache/fence operations do not need a separate device-cache model.
    if op in {"fence", "icinv", "dccln", "dcinv", "dcflu", "ccinv", "sleep", "nop"}: return UOp.sink()
    if op in {"lock", "unlock"}: raise NotImplementedError(f"{name} needs shared atomic/lock semantics")
    raise RuntimeError(f"Unsupported category 7 instruction: {name!r}")


def _expand_repeat(inst: "Inst") -> "list[Inst]":
    """(rptN) on cat1-4 runs the instruction N+1 times, in order. Each run steps the destination and every (r)-flagged source
    register by one more component (w carries into the next register's x); sources without (r) are re-read unchanged (ir3.h:
    `(rpt2)mad.f32 r3.y, (r)c9.x, r1.x, (r)r3.z`). On cat0 the same field is the (nopN) count, so nothing repeats there."""
    if inst.cat not in {1, 2, 3, 4} or not inst.fields: return [inst]
    rpt = [int(m[3:]) for m in inst.fields[0].modifiers if re.fullmatch(r"rpt\d+", m)]
    if not rpt or _inst_name(inst).strip().lower() == "nop": return [inst]
    name = Field(inst.fields[0].value, tuple(m for m in inst.fields[0].modifiers if m != f"rpt{rpt[0]}"))
    ret = []
    for i in range(rpt[0] + 1):
        fields = [name]
        for j, f in enumerate(inst.fields[1:], 1):
            value = f.value
            if _QCOM_REG_RE.fullmatch(value.strip()) and (j == 1 or "r" in f.modifiers): value = _qcom_reg_component_offset_token(value, i)
            fields.append(Field(value, tuple(m for m in f.modifiers if m != "r")))
        ret.append(Inst(inst.cat, tuple(fields)))
    return ret

def _compile_inst(inst, ctx):
    if len(_expand_repeat(inst)) > 1: raise RuntimeError(f"(rptN) must be expanded with _expand_repeat before compiling: {inst!r}")
    if inst.cat == 0:
        return _compile_cat0(inst, ctx)
    elif inst.cat == 1:
        return _compile_cat1(inst, ctx)
    elif inst.cat == 2:
        return _compile_cat2(inst, ctx)
    elif inst.cat == 3:
        return _compile_cat3(inst, ctx)
    elif inst.cat == 4:
        return _compile_cat4(inst, ctx)
    elif inst.cat == 5:
        return _compile_cat5(inst, ctx)
    elif inst.cat == 6:
        return _compile_cat6(inst, ctx)
    elif inst.cat == 7:
        return _compile_cat7(inst, ctx)
    else: raise RuntimeError(f"Invalid instruction category: {inst.cat}")

def _end_lane_range(sink):
    # every compile function builds its per-lane work inside one ctx.range(); close it here (like the AMD emulator's .end(lane))
    rngs = list(sink.ranges)  # the still-open ranges: earlier instructions of a fused block are already ended
    assert len(rngs) <= 1, f"expected at most one lane range, got {len(rngs)}"
    return UOp.sink(sink.end(*rngs)) if rngs else sink

# Lowering a run of instructions to a CPU kernel costs tens of ms per instruction in graph rewrites and a real kernel has hundreds of
# them, so lowered programs are kept on disk, keyed by the instructions and by the part of this file that decides what a kernel computes
# (NUM_R_REG up to WaveState; editing the imports or the scheduler does not invalidate them).
_src = pathlib.Path(__file__).read_text()
_EMU_SRC_HASH = hashlib.sha256(_src[_src.index("NUM_R_REG"):_src.index("class WaveState")].encode()).hexdigest()  # not the imports above it

MAX_BLOCK = int(os.environ.get("QCOM_EMU_MAX_BLOCK", "48"))  # longest run of instructions fused into one kernel (compile time grows faster than linearly past this)

def _block_key(insts: tuple, merged: bool) -> dict:
    return {"emu": _EMU_SRC_HASH, "insts": hashlib.sha256(repr(insts).encode()).hexdigest(), "merged": int(merged)}

def _is_lane_local(inst: "Inst") -> bool:
    """ALU-style instructions only touch their own lane's registers, so a run of them can share one loop over the lanes."""
    return inst.cat in {1, 2, 3, 4} and not _inst_name(inst).startswith(("swz", "gat", "sct", "mova", "movs", "movmsk"))


def _lower_block(insts: tuple, merged: bool, renderer=None):
    """Lower a straight-line run of instructions (each (rptN) already expanded) to one CPU program; False if there is no work."""
    with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
        ctx = _Ctx(merged=merged)
        sinks: list[UOp] = []
        group: list[UOp] = []  # consecutive lane-local instructions share one loop over the lanes (the values stay in registers in between)

        def flush():
            if not group: return
            sinks.append(sink:=_end_lane_range(UOp.sink(*group)))
            ctx.advance(sink)
            group.clear()
            ctx.fused_lane = None
        for inst in insts:
            if _is_lane_local(inst):
                if ctx.fused_lane is None: ctx.fused_lane = ctx.range()  # the first instruction of a group opens the loop
                sink = _compile_inst(inst, ctx)
                if sink.op is Ops.SINK and not sink.src: continue
                group.append(sink)
                ctx.advance(sink)  # later instructions of the loop read what this one wrote (same lane, so only ordering is needed)
            else:
                flush()
                ctx.fused_lane = None
                sink = _end_lane_range(_compile_inst(inst, ctx))
                if sink.op is Ops.SINK and not sink.src: continue  # nop-like: nothing to run
                sinks.append(sink)
                ctx.advance(sink)
        flush()
        if not sinks: return False
        return to_program(UOp.sink(*sinks).replace(arg=KernelInfo(name=f"qcom_{_inst_name(insts[0])}_x{len(insts)}")), renderer or Device['CPU'].renderer)

_lower_block_task = lambda args: (args[0], _lower_block(*args[1:]))  # noqa: E731  (pool task: worker processes cannot open devices, so the renderer is passed in)

_PROGRAMS: dict[tuple[str, bool], Any] = {}  # lowered programs seen by this process, by key

def _load_program(insts: tuple, merged: bool, lower: bool = True):
    key = _block_key(insts, merged)
    if (prg:=_PROGRAMS.get((key["insts"], merged))) is None and (prg:=diskcache_get("qcom_emu_block", key)) is None and lower:
        prg = diskcache_put("qcom_emu_block", key, _lower_block(insts, merged))
    if prg is not None: _PROGRAMS[(key["insts"], merged)] = prg
    return prg

@functools.cache
def _get_block_runner(insts: tuple, merged=False):
    """The runnable kernel for a straight-line run of instructions, or None if the run has no work."""
    prg = _load_program(insts, merged)
    with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
        return (prg, get_runtime('CPU', prg)) if prg is not False else None

def _precompile(blocks: list[tuple], merged: bool) -> None:
    """Lower every block of a shader that is not cached yet, in tinygrad's worker pool (a cold shader has hundreds)."""
    todo = {k["insts"]: b for b in blocks if _load_program(b, merged, lower=False) is None and (k:=_block_key(b, merged))}
    if len(todo) < 6 or (pool:=get_worker_pool()) is None: return  # not worth the round trips / no workers: compiled when first run
    renderer, keys = Device['CPU'].renderer, list(todo)
    try:
        for i, prg in pool.imap_unordered(_lower_block_task, [(i, todo[k], merged, renderer) for i, k in enumerate(keys)]):
            diskcache_put("qcom_emu_block", _block_key(todo[keys[i]], merged), prg)
            _PROGRAMS[(keys[i], merged)] = prg
    except Exception: pass  # an unsupported instruction: it is reported with its name if the block is ever run

def _compile_run(pc, decoded_shader, n, merged=False):
    """Compile instructions [pc, pc+n) as one kernel; the error names the instruction that failed."""
    insts = tuple(i for inst in decoded_shader[pc:pc+n] for i in _expand_repeat(inst))
    try: return _get_block_runner(insts, merged)
    except Exception:
        for inst in insts:  # find the culprit for the message
            try: _end_lane_range(_compile_inst(inst, _Ctx(merged=merged)))
            except Exception as e: raise RuntimeError(f"[emu] Failed to compile {inst!r}: {type(e).__name__}: {e}") from e
        raise


class WaveState:
    __slots__ = ("wave_size", "wave_idx", "r_reg", "s_reg", "c_reg", "a_reg", "p_reg", "_r_reg_mv", "_s_reg_mv", "_c_reg_mv", "_hr_reg_mv", "_hs_reg_mv", "_hc_reg_mv", "_a_reg_mv", "_p_reg_mv", "pc", "exec_mask", "exec_mask_buf", "_exec_mask_mv", "lane_mask", "predication_mask", "predication_mode", "barrier_waiting", "call_stack", "mergedregs", "parked", "launch_mask")

    def __init__(
        self, wave_size, wave_idx,
        shader_constants, wgidconstid, wgsizeconstid, localidregid,
        wgid, wgsize, local_dims, mergedregs=False
    ):
        self.wave_size = wave_size
        self.mergedregs = mergedregs
        self.wave_idx = wave_idx
        active_lanes = min(wave_size, max(0, math.prod(local_dims) - wave_idx * wave_size))
        self.launch_mask = (1 << active_lanes) - 1
        self.lane_mask = (1 << active_lanes) - 1  # lanes currently running; lanes waiting at a reconvergence point are removed (see parked)
        self.parked: dict[int, int] = {}  # pc -> lanes parked there by a divergent branch, rejoined when the wave reaches that pc
        self.exec_mask = self.lane_mask
        self.exec_mask_buf = Buffer('CPU', 8).ensure_allocated()
        self._exec_mask_mv = self.exec_mask_buf.host.view(fmt='Q').mv
        self._exec_mask_mv[0] = self.exec_mask
        self.predication_mask = 0
        self.predication_mode = "none"
        self.barrier_waiting = False
        self.call_stack: list[int] = []

        #r0 - r47
        self.r_reg = Buffer('CPU', R_WORDS * self.wave_size * 4)
        self._r_reg_mv = self.r_reg.host.view(fmt='I').mv
        self._hr_reg_mv = self.r_reg.host.view(fmt='H').mv
        #shared across threads
        #r48 - r55
        self.s_reg = Buffer('CPU', NUM_S_REG * 4)
        self._s_reg_mv = self.s_reg.host.view(fmt='I').mv
        self._hs_reg_mv = self.s_reg.host.view(fmt='H').mv

        self.c_reg = Buffer('CPU', NUM_C_REG * 4)
        self._c_reg_mv = self.c_reg.host.view(fmt='I').mv
        self._hc_reg_mv = self.c_reg.host.view(fmt='H').mv

        self.a_reg = Buffer('CPU', NUM_A_REG * 4)
        self._a_reg_mv = self.a_reg.host.view(fmt='I').mv
        # Back p0.x/y/z/w with a 0/1 word per lane. This makes predicate destinations
        # straightforward to compile as vector UOps; _read_p_reg packs them for control flow.
        self.p_reg = Buffer('CPU', NUM_P_REG * self.wave_size * 4)
        self._p_reg_mv = self.p_reg.host.view(fmt='I').mv

        ctypes.memset(self.r_reg._buf, 0, R_WORDS * self.wave_size * 4)
        ctypes.memset(self.s_reg._buf, 0, NUM_S_REG * 4)
        ctypes.memset(self.c_reg._buf, 0, NUM_C_REG * 4)
        ctypes.memset(self.a_reg._buf, 0, NUM_A_REG * 4)
        ctypes.memset(self.p_reg._buf, 0, NUM_P_REG * wave_size * 4)

        self.pc = 0

        # LOAD_STATE6 constants are supplied as 32-bit words; populate cN.{x,y,z,w}.
        # System values are written afterwards so their designated slots take precedence.
        for i, val in enumerate(shader_constants[:NUM_C_REG]):
            self._write_c_reg(i // 4, i % 4, int(val))

        # Each *regid is (register << 2 | first component) with 0xFC meaning "unused" (Mesa INVALID_REG); a vec3 occupies
        # three consecutive components, which may start mid-register (ir3_find_sysval_regid / create_sysval_input mask 0x7).
        # (tinygrad's shaders read the workgroup id from r48.x/y/z, i.e. the shared file, so a register id >= 48 lands there)
        self._write_wgid(wgidconstid, wgid)
        if wgsizeconstid != 0xFC:
            for i in range(3):
                self._write_c_reg((wgsizeconstid + i) >> 2, (wgsizeconstid + i) & 3, wgsize[i])

        active_lanes = min(self.wave_size, max(0, math.prod(local_dims) - wave_idx * self.wave_size))
        if localidregid != 0xFC:
            for i in range(active_lanes):
                global_pos = i + wave_idx * wave_size
                local_x = global_pos % local_dims[0]
                local_y = (global_pos // local_dims[0]) % local_dims[1]
                local_z = global_pos // (local_dims[0] * local_dims[1])
                localid = (local_x, local_y, local_z)
                for j in range(3):
                    self._write_r_reg(i, (localidregid + j) >> 2, (localidregid + j) & 3, localid[j])
        #wgoffset and linearlocalid not exposed

    def _decode_reg(self, reg_id):
        return reg_id >> 2, reg_id & 3

    def _write_r_reg(self, thread_id, reg_num, reg_dim, val, half=False):
        if half:
            self._hr_reg_mv[thread_id * R_WORDS * 2 + (0 if self.mergedregs else NUM_R_REG * 2) + reg_num * 4 + reg_dim] = val
        else:
            self._r_reg_mv[thread_id * R_WORDS + reg_num * 4 + reg_dim] = val
    def _read_r_reg(self, thread_id, reg_num, reg_dim, half=False):
        if half:
            return self._hr_reg_mv[thread_id * R_WORDS * 2 + (0 if self.mergedregs else NUM_R_REG * 2) + reg_num * 4 + reg_dim]
        else:
            return self._r_reg_mv[thread_id * R_WORDS + reg_num * 4 + reg_dim]
    def _write_s_reg(self, reg_num, reg_dim, val, half=False):
        if half:
            self._hs_reg_mv[reg_num * 4 + reg_dim] = val
        else:
            self._s_reg_mv[reg_num * 4 + reg_dim] = val
    def _read_s_reg(self, reg_num, reg_dim, half=False):
        if half:
            return self._hs_reg_mv[reg_num * 4 + reg_dim]
        else:
            return self._s_reg_mv[reg_num * 4 + reg_dim]
    #only during initialisation (const registers)
    def _write_c_reg(self, reg_num, reg_dim, val, half=False):
        if half:
            self._hc_reg_mv[reg_num * 4 + reg_dim] = val
        else:
            self._c_reg_mv[reg_num * 4 + reg_dim] = val
    def _read_c_reg(self, reg_num, reg_dim, half=False):
        if half:
            return self._hc_reg_mv[reg_num * 4 + reg_dim]
        else:
            return self._c_reg_mv[reg_num * 4 + reg_dim]
    def _write_a_reg(self, reg_num, val):
        self._a_reg_mv[reg_num] = val
    def _read_a_reg(self, reg_num):
        return self._a_reg_mv[reg_num]
    def _write_p_reg(self, reg_num, reg_dim, val):
        # Scalar helper used by setup/low-level code: write mask bits to per-lane booleans.
        for lane in range(self.wave_size):
            self._p_reg_mv[lane * NUM_P_REG + reg_num * 4 + reg_dim] = (val >> lane) & 1
    def _read_p_reg(self, reg_num, reg_dim):
        # one 0/1 word per lane, NUM_P_REG words apart: take that word's low byte for every lane and read the bytes as binary digits
        raw = ctypes.string_at(self.p_reg._buf, self.wave_size * NUM_P_REG * 4)
        return int(raw[(reg_num * 4 + reg_dim) * 4::NUM_P_REG * 4].translate(_BIT_DIGITS)[::-1], 2)

    def _write_wgid(self, wgidconstid, wgid):
        # (tinygrad's shaders read the workgroup id from r48.x/y/z, i.e. the shared file, so a register id >= 48 lands there)
        if wgidconstid == 0xFC: return
        for i in range(3):
            reg = (wgidconstid + i) >> 2
            if reg >= 48: self._write_s_reg(reg - 48, (wgidconstid + i) & 3, wgid[i])
            else: self._write_c_reg(reg, (wgidconstid + i) & 3, wgid[i])

    def snapshot(self) -> tuple[bytes, bytes, bytes]:
        """The register images a fresh wave starts from (everything but the workgroup id is the same for every workgroup)."""
        return (ctypes.string_at(self.r_reg._buf, R_WORDS * self.wave_size * 4), ctypes.string_at(self.c_reg._buf, NUM_C_REG * 4),
                ctypes.string_at(self.s_reg._buf, NUM_S_REG * 4))

    def reset(self, images: tuple[bytes, bytes, bytes], wgidconstid, wgid) -> None:
        """Return to the state at the start of a workgroup without reallocating anything."""
        for buf, img in zip((self.r_reg, self.c_reg, self.s_reg), images): ctypes.memmove(buf._buf, img, len(img))
        ctypes.memset(self.a_reg._buf, 0, NUM_A_REG * 4)
        ctypes.memset(self.p_reg._buf, 0, NUM_P_REG * self.wave_size * 4)
        self._write_wgid(wgidconstid, wgid)
        self.lane_mask = self.exec_mask = self.launch_mask
        self._exec_mask_mv[0] = self.exec_mask
        self.parked = {}
        self.predication_mask, self.predication_mode, self.barrier_waiting, self.call_stack, self.pc = 0, "none", False, [], 0

def _init_wave(
    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
    wave_start, wave_size, total_threads, local_dims, pvt_memsize_peritem,
    wgidconstid, wgsizeconstid, localidregid, group_id, group_dims, mergedregs=False
):
    return WaveState(
        wave_size=wave_size,
        wave_idx=wave_start // wave_size,
        shader_constants=shader_constants,
        wgidconstid=wgidconstid,
        wgsizeconstid=wgsizeconstid,
        localidregid=localidregid,
        wgid=group_id,
        wgsize=local_dims,
        local_dims=local_dims,
        mergedregs=mergedregs
    )

def _int_to_byte_arr(arr):
    ret = []
    for word in arr:
        for i in range(4):
            ret.append((word>>(i*8)) & 0xFF)
    return ret


def _disassemble_shader(lib:Any, gpu_id=630):
  with tempfile.TemporaryFile('w+') as tf:
    mesa_fp = ctypes.cast(fp:=libc.fdopen(tf.fileno(), b"w"), ctypes.POINTER(mesa.struct__IO_FILE))
    mesa.ir3_isa_disasm(lib, len(lib), mesa_fp, mesa.struct_isa_decode_options(gpu_id, True, 0, True))
    libc.fflush(fp)

    tf.seek(0)
    return tf.read()

@dataclass(frozen=True)
class Field:
    value: str
    modifiers: tuple[str, ...]

@dataclass(frozen=True)
class Inst:
    cat: int
    fields: tuple[Field, ...] # even the name is just the first field


def _inst_name(inst: Inst) -> str:
    return inst.fields[0].value if inst.fields else ""


def _is_end_inst(inst: Inst) -> bool:
    return inst.cat == 0 and _inst_name(inst).strip().lower() == "end"


def _is_barrier_inst(inst: Inst) -> bool:
    return inst.cat == 7 and _inst_name(inst).strip().lower().split(".", 1)[0] == "bar"


def _is_predication_enable_inst(inst: Inst) -> bool:
    return inst.cat == 0 and _inst_name(inst) in {"predt", "predf"}


def _is_predication_disable_inst(inst: Inst) -> bool:
    return inst.cat == 0 and _inst_name(inst) == "prede"


def _combine_predicated_exec_mask(exec_mask: int, predication_mask: int, mode: str, width: int | None = None) -> int:
    if width is None: width = max(exec_mask.bit_length(), predication_mask.bit_length(), 1)
    full_mask = (1 << width) - 1
    exec_mask &= full_mask
    predication_mask &= full_mask
    if mode == "none": return exec_mask
    if mode == "true": return exec_mask & predication_mask
    if mode == "false": return exec_mask & (~predication_mask & full_mask)
    raise ValueError(f"unknown predication mode: {mode!r}")


_QCOM_INSTRUCTION_CONTROL_FLAGS = {"jp", "sy", "ss", "eq", "eolm", "eogm", "eostsc", "sat"}
_QCOM_RESULT_FLAGS = {"sat"}
_QCOM_REGISTER_QUALIFIERS = {"r", "rel"}
_QCOM_TEX_TYPES = {"f32", "f16", "u32", "u16", "s32", "s16", "u8"}  # cat5 "(f32)(xyzw)r2.y": data type and component write mask of the result
_QCOM_OPERAND_VALUE_MODIFIERS = {"neg", "n", "abs", "a", "bnot", "not"}
# the float lookup-table immediates of ir3-common.xml (#flut); the disassembler prints the ones that are not plain numbers by name
_QCOM_FLUT_IMMS = {"e": math.e, "pi": math.pi, "1/pi": 1/math.pi, "1/log2(e)": 1/math.log2(math.e), "log2(e)": math.log2(math.e),
                   "1/log2(10)": 1/math.log2(10), "log2(10)": math.log2(10)}
_QCOM_FLOAT_IMM_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+e[+-]?\d+|inf|nan)", re.IGNORECASE)


def _split_qcom_modifiers(token: str) -> tuple[str, list[str]]:
    """Return (value-token, actual operand modifiers) for the A6XX ISA.

    Only true value modifiers survive: (neg), (abs), (not)/ (bnot). Header/control flags
    such as (jp), (sy), (ss), (eq), and encoding qualifiers such as (r) are intentionally ignored.
    """
    token = token.strip()
    if not token:
        return "", []
    expr = token
    mods: list[str] = []
    while True:
        match = re.match(r"^\(([^()]+)\)", expr)
        if match is None:
            break
        mod = match.group(1).strip().lower()
        # The Mesa disassembler prints float immediates in parentheses, e.g. "add.f r2.w, r2.w, (1.0)".
        if (_QCOM_FLOAT_IMM_RE.fullmatch(mod) or mod in _QCOM_FLUT_IMMS) and match.end() == len(expr): return mod, mods
        if mod in _QCOM_OPERAND_VALUE_MODIFIERS or mod in _QCOM_RESULT_FLAGS or mod == "r" or re.fullmatch(r"rpt\d+", mod) or mod in _QCOM_TEX_TYPES or re.fullmatch(r"[xyzw]{1,4}", mod):
            mods.append(mod)
        expr = expr[match.end():].strip()
    return expr, mods


def _apply_qcom_operand_modifiers(value: UOp, modifiers: list[str] | tuple[str, ...] | None) -> UOp:
    for mod in modifiers or []:
        mod = mod.strip().lower()
        if mod in _QCOM_REGISTER_QUALIFIERS or mod in _QCOM_INSTRUCTION_CONTROL_FLAGS:
            continue
        if mod in {"abs", "a"}:
            # NOTE: abs of an unsigned view is a no-op; callers that mean a signed/float abs must decode the operand first
            # (_qcom_float_operand / _qcom_read_typed_operand do). Generic raw reads also pass through here unused.
            value = value.abs()
        elif mod in {"neg", "n"}:
            if value.dtype in {dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64}:
                if value.dtype in {dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64}:
                    value = (~value + _c(1, value.dtype)).cast(value.dtype)
                else:
                    value = (-value).cast(value.dtype)
            else:
                value = (-value).cast(value.dtype)
        elif mod in {"bnot", "not"}:
            if value.dtype in {dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64}:
                value = (~value).cast(value.dtype)
            else:
                value = (~value.cast(dtypes.uint32)).cast(value.dtype)
    return value


def _parse_line(line):
    # Disassembly formatting can contain leading or repeated whitespace. split() also
    # prevents empty operands from being introduced by repeated spaces or tabs.
    fields: list[Field] = []
    pending: list[str] = []
    for word in line.replace(",", "").split():
        value, modifiers = _split_qcom_modifiers(word)
        # A word made only of flag groups, e.g. "(sy)(nop3)" or a lone "(r)", carries no operand: keep its value modifiers
        # (sat/rptN) for the next field instead of emitting an empty field that would shift every operand.
        if not value:
            pending += modifiers
            continue
        fields.append(Field(value, tuple(pending + modifiers)))
        pending = []
    return fields


def _parse_instructions(disassembled_shader):
    # Branch labels refer to instruction PCs, not physical lines in the disassembly:
    # labels are removed from the instruction stream, so counting source lines shifts
    # every target after the first label.
    instructions: list[list[Field]] = []
    jump_locations: dict[str, int] = {}
    for raw_line in disassembled_shader.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            if label in jump_locations:
                raise RuntimeError(f"duplicate IR3 branch label {label!r}")
            jump_locations[label] = len(instructions)
            continue
        fields = _parse_line(line)
        if fields:
            instructions.append(fields)
    return instructions, jump_locations

def _extract_cat(inst):
    return (inst >> 61) & 0x7

def _decode_program(shader):
    # `shader` is a sequence of 32-bit words; each IR3 instruction is two words.
    if len(shader) % 2:
        raise ValueError(f"IR3 shader has an odd number of 32-bit words: {len(shader)}")
    shader_buffer = bytearray(_int_to_byte_arr(shader))
    shader_arr = (ctypes.c_char * len(shader_buffer)).from_buffer(shader_buffer)

    disassembled_shader = _disassemble_shader(shader_arr)
    parsed_list, jump_locations = _parse_instructions(disassembled_shader)
    expected_instructions = len(shader) // 2
    if len(parsed_list) != expected_instructions:
        raise RuntimeError(
            f"IR3 disassembler produced {len(parsed_list)} instructions for {expected_instructions} encoded instructions"
        )
    instructions = [
        # Category occupies bits 61..63 of the 64-bit instruction, i.e. the top
        # three bits of the second (high) 32-bit word in the little-endian word stream.
        Inst(_extract_cat((int(shader[2*i + 1]) << 32) | int(shader[2*i])), tuple(parsed_list[i]))
        for i in range(expected_instructions)
    ]
    return instructions, jump_locations

def _apply_wave_predication(inst: Inst, state: WaveState) -> None:
    """Apply A6xx predt/predf/prede state to the wave execution mask."""
    name = _inst_name(inst).lower()
    full = (1 << state.wave_size) - 1
    if name in {"predt", "predf"}:
        # Mesa's IR3 ISA notes specify that predt/predf snapshot p0.x, not the
        # current exec mask. The original physical mask is applied separately.
        state.predication_mask = state._read_p_reg(0, 0) & full
        state.predication_mode = "true" if name == "predt" else "false"
        state.exec_mask = _combine_predicated_exec_mask(state.lane_mask, state.predication_mask,
                                                        state.predication_mode, state.wave_size)
    elif name == "prede":
        state.predication_mode = "none"
        state.exec_mask = state.lane_mask
    state._exec_mask_mv[0] = state.exec_mask & 0xFFFFFFFFFFFFFFFF


def _is_cat0_branch(inst: Inst) -> bool:
    return inst.cat == 0 and _inst_name(inst).strip().lower() in {"jump", "call", "ret", "br", "bany", "ball", "brao", "braa"}


def _branch_target(token: str, pc: int, jump_locations: dict[str, int], shader_len: int) -> int:
    """Resolve a disassembler label, or a signed instruction-relative branch displacement."""
    token = token.strip().lstrip("#")
    label = re.fullmatch(r"(?:l|fxn)(\d+)", token, re.IGNORECASE)
    if token in jump_locations:
        # Resolve known labels first: a label such as l0 is not necessarily instruction 0.
        target = jump_locations[token]
    elif label is not None:
        target = int(label.group(1))
    else:
        # Per Mesa ISASpec's branch-field contract, a `branch` immediate is a signed
        # instruction-relative displacement; IR3 instructions are fixed 64-bit slots.
        try:
            offset = int(token, 0)
        except ValueError as e:
            raise NotImplementedError(f"Cannot resolve IR3 branch target {token!r}; known labels: {tuple(jump_locations)}") from e
        # Textual disassembly may already render a negative signed displacement (#-4).
        # Only sign-extend values that are still represented as unsigned 32-bit words.
        if 0 <= offset <= 0xFFFFFFFF and offset & 0x80000000: offset -= 1 << 32
        target = pc + offset
    if not 0 <= target < shader_len:
        raise RuntimeError(f"IR3 branch from PC {pc} resolves to out-of-range target {target} (shader has {shader_len} instructions)")
    return target


def _predicate_mask(state: WaveState, token: str) -> int:
    token = token.strip()
    invert = token.startswith("!")
    if invert: token = token[1:]
    kind, reg, comp = _qcom_reg_token(token)
    if kind != "p": raise NotImplementedError(f"Cat0 branch predicate must be pN.component, got {token!r}")
    value = state._read_p_reg(reg, comp) & state.lane_mask
    return ((~value) & state.lane_mask) if invert else value


def _branch_info(inst: Inst, pc: int, jump_locations: dict[str, int], shader_len: int) -> tuple:
    """Everything a cat0 branch needs, parsed once: (name, target, ((invert, p register, component), ...))."""
    name = _inst_name(inst).strip().lower()
    if name in {"call", "jump"} and len(inst.fields) < 2: raise RuntimeError(f"IR3 {name} missing target: {inst!r}")
    target = _branch_target(inst.fields[-1].value, pc, jump_locations, shader_len) if name != "ret" else -1
    if name in {"ret", "call", "jump"}: return name, target, ()
    preds = []
    for tok in (f.value for f in inst.fields[1:-1] if f.value.lstrip("!").startswith("p")):
        kind, reg, comp = _qcom_reg_token(tok.lstrip("!"))
        if kind != "p": raise NotImplementedError(f"Cat0 branch predicate must be pN.component, got {tok!r}")
        preds.append((tok.startswith("!"), reg, comp))
    if not preds: raise RuntimeError(f"IR3 {name} missing predicate operand(s): {inst!r}")
    if name in {"brao", "braa"} and len(preds) != 2: raise RuntimeError(f"IR3 {name} expects two predicates: {inst!r}")
    if name not in {"br", "bany", "ball", "brao", "braa"}: raise NotImplementedError(f"IR3 branch {name!r} is not implemented")
    return name, target, tuple(preds)


def _execute_cat0_branch(info: tuple, state: WaveState, pc: int) -> int:
    """Execute the branch subset needed by compute shaders; return the next PC."""
    name, target, preds = info
    if name == "ret":
        if not state.call_stack: raise RuntimeError(f"IR3 ret at PC {pc} with an empty call stack")
        return state.call_stack.pop()
    if name == "call":
        state.call_stack.append(pc + 1)
        return target
    if name == "jump": return _goto(state, pc, target)

    # a6xx divergence model (ir3_reconvergence.c): lanes that take a branch to a later block are "parked" there while the others
    # carry on; parked lanes rejoin when the wave reaches that pc (the (jp) reconvergence point). For a backward branch the lanes
    # that leave the loop park at the next instruction instead. Predication (predt/predf) never parks lanes.
    masks = [(state._read_p_reg(reg, comp) ^ (state.launch_mask if invert else 0)) & state.launch_mask for invert, reg, comp in preds]
    active = state.exec_mask & state.lane_mask
    if name == "br": taken = masks[0] & active
    elif name == "bany": taken = active if masks[0] & active else 0  # bany/ball reduce over the wave, so they never diverge
    elif name == "ball": taken = active if active and (masks[0] & active) == active else 0
    else: taken = ((masks[0] | masks[1]) if name == "brao" else (masks[0] & masks[1])) & active
    if taken == active: return _goto(state, pc, target)
    if not taken: return pc + 1
    if target > pc: _park(state, target, taken)  # forward: the taken lanes wait at the target
    else: _park(state, pc + 1, active & ~taken)  # backward: the lanes leaving the loop wait after it
    return pc + 1 if target > pc else target


def _refresh_exec_mask(state: WaveState) -> None:
    state.exec_mask = state.lane_mask if state.predication_mode == "none" else _combine_predicated_exec_mask(
        state.lane_mask, state.predication_mask, state.predication_mode, state.wave_size)
    state._exec_mask_mv[0] = state.exec_mask & 0xFFFFFFFFFFFFFFFF


def _park(state: WaveState, pc: int, lanes: int) -> None:
    state.parked[pc] = state.parked.get(pc, 0) | lanes
    state.lane_mask &= ~lanes
    _refresh_exec_mask(state)


def _goto(state: WaveState, pc: int, target: int) -> int:
    """All running lanes go to target. A forward jump over a pc where lanes are parked cannot skip it: the lanes park at the
    target and the wave falls through (with nothing running) to the parked pc."""
    if target > pc and any(pc < k < target for k in state.parked):
        _park(state, target, state.lane_mask)
        return pc + 1
    return target


def _unpark(state: WaveState, pc: int) -> None:
    state.lane_mask |= state.parked.pop(pc)
    _refresh_exec_mask(state)


def _dump_lane0(st: WaveState) -> str:
    full = {f"r{i // 4}.{'xyzw'[i % 4]}": st._r_reg_mv[i] for i in range(NUM_R_REG) if st._r_reg_mv[i]}
    half_base = 0 if st.mergedregs else NUM_R_REG * 2
    half = {f"hr{i // 4}.{'xyzw'[i % 4]}": st._hr_reg_mv[half_base + i] for i in range(NUM_R_REG * 2) if st._hr_reg_mv[half_base + i]}
    return " ".join(f"{k}={v:#x}" for k, v in {**full, **half}.items())


def _barrier_resume_ready(waves: list[WaveState]) -> bool:
    if not waves: return False
    return all(w.barrier_waiting for w in waves)


def _descriptor_buffer(words, what: str) -> Buffer:
    """Descriptors as loaded by CP_LOAD_STATE6 (16 dwords each) in a buffer the kernels can read. Only what the instructions above model
    is accepted: linear 2D RGBA32F / RGBA16F images; anything else would be silently misread, so it is refused up front."""
    words = list(words)[:16 * NUM_IMG_DESC]
    buf = Buffer('CPU', 16 * NUM_IMG_DESC * 4).ensure_allocated()
    ctypes.memset(buf._buf, 0, 16 * NUM_IMG_DESC * 4)
    for i, w in enumerate(words): ctypes.c_uint32.from_address(buf._buf + 4 * i).value = int(w)
    for d in range(len(words) // 16):
        d0, d2, d4 = words[16 * d], words[16 * d + 2], words[16 * d + 4]
        if not d4 and not words[16 * d + 5]: continue  # unused slot
        fmt = (d0 & mesa.A6XX_TEX_CONST_0_FMT__MASK) >> mesa.A6XX_TEX_CONST_0_FMT__SHIFT
        if (d0 & mesa.A6XX_TEX_CONST_0_TILE_MODE__MASK) != 0 or (d2 & mesa.A6XX_TEX_CONST_2_TYPE__MASK) >> mesa.A6XX_TEX_CONST_2_TYPE__SHIFT != mesa.A6XX_TEX_2D or \
           fmt not in {mesa.FMT6_32_32_32_32_FLOAT, mesa.FMT6_16_16_16_16_FLOAT}:
            raise NotImplementedError(f"{what} descriptor {d} is not a linear 2D RGBA32F/RGBA16F image: {[hex(int(w)) for w in words[16 * d:16 * d + 6]]}")
    return buf


def run_asm(
    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
    global_dims, group_dims, local_dims, kerneldim, wave_size,
    shared_memsize, pvt_memsize_peritem, pvt_mem_base,
    wgidconstid, wgsizeconstid, wgoffsetconstid, localidregid, linearlocalidregid, mergedregs=False
):
    if wave_size != 64:
        raise NotImplementedError(f"QCOM UOp runner currently supports wave64 only, got wave_size={wave_size}")
    decoded_shader, jump_locations = _decode_program(shader)

    program: dict[int, tuple[bool, Any, bool, bool, bool, Inst, int]] = {}

    total_threads = math.prod(local_dims)

    vmem_buf = Buffer("CPU", 1<<48, options=BufferSpec(external_ptr=0)).ensure_allocated()
    pvt_bytes_per_lane = pvt_memsize_peritem * 512  # MEMSIZEPERITEM is in 512-byte units (ops_qcom: round_up(pvtmem, 512) >> 9)

    n_inst = len(decoded_shader)
    # Instructions that need the scheduler: branches, predication, barriers and the end. Everything between them is fused into runs.
    special = [_is_cat0_branch(i) or _is_predication_enable_inst(i) or _is_predication_disable_inst(i) or _is_barrier_inst(i) or _is_end_inst(i) for i in decoded_shader]
    leaders = {0} | {_branch_target(i.fields[-1].value, pc, jump_locations, n_inst) for pc, i in enumerate(decoded_shader) if _is_cat0_branch(i) and i.fields and _inst_name(i) != "ret"}
    def _run_len(pc):  # a run stops before a branch target (lanes can arrive there from elsewhere) and before anything special
        n = 1
        while pc + n < n_inst and n < MAX_BLOCK and not special[pc + n] and (pc + n) not in leaders: n += 1
        return n
    blocks, pc = [], 0
    while pc < n_inst:  # the runs the scheduler will execute, lowered up front (in parallel) when they are not cached
        n = 1 if special[pc] else _run_len(pc)
        if not special[pc]: blocks.append(tuple(i for inst in decoded_shader[pc:pc + n] for i in _expand_repeat(inst)))
        pc += n
    _precompile(blocks, mergedregs)
    trace_regs = DEBUG >= 6  # also dump lane 0's registers after every instruction
    stats = [0.0, 0] if os.environ.get("QCOM_EMU_STATS") else None  # [seconds inside compiled kernels, kernel calls], printed when run_asm returns
    trace = DEBUG >= 5  # read once: ContextVar comparisons are slow in a loop that runs once per instruction

    def _ensure_compiled(pc):
        # what the scheduler needs to know about an instruction, worked out once (this runs for every instruction of every wave)
        if pc not in program:
            inst = decoded_shader[pc]
            if _is_cat0_branch(inst): program[pc] = (True, _branch_info(inst, pc, jump_locations, n_inst), False, False, False, inst, 1)
            else:
                n = 1 if special[pc] else _run_len(pc)
                prev_len = _get_block_runner.cache_info().currsize
                runner = _compile_run(pc, decoded_shader, n, mergedregs)
                program[pc] = (False, [(runner[1].fxn, runner[0].arg.globals)] if runner is not None else [], _is_barrier_inst(inst), _is_end_inst(inst),
                               _is_predication_enable_inst(inst) or _is_predication_disable_inst(inst), inst, n)
                if DEBUG >= 3:
                    msg = f"[emu] PC={pc}..{pc + n - 1}: {inst!r}"
                    print(colored(msg, 'green') if _get_block_runner.cache_info().currsize > prev_len else msg)
        return program[pc]

    shared_buf = Buffer('CPU', max(shared_memsize, 1)).ensure_allocated()
    tex_buf, ibo_buf = _descriptor_buffer(shader_tex_constants, "texture"), _descriptor_buffer(shader_uav, "IBO")
    pool: list[Any] = []  # the waves of a workgroup are the same objects for every workgroup, just reset: (state, buffers, kernel args, images)

    def _run_workgroup(group_id):
        ctypes.memset(shared_buf._buf, 0, max(shared_memsize, 1))
        if not pool:
            for wave_start in range(0, total_threads, wave_size):
                st = _init_wave(
                    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
                    wave_start, wave_size, total_threads, local_dims, pvt_memsize_peritem,
                    wgidconstid, wgsizeconstid, localidregid, group_id, group_dims, mergedregs
                )
                pvt_buf = Buffer('CPU', max(pvt_bytes_per_lane * wave_size, 1)).ensure_allocated()
                tmp_buf = Buffer('CPU', 4 * wave_size * 4).ensure_allocated()
                pool.append((st,
                    [
                        ctypes.c_uint64(st.r_reg._buf),
                        ctypes.c_uint64(st.s_reg._buf),
                        ctypes.c_uint64(st.c_reg._buf),
                        ctypes.c_uint64(st.a_reg._buf),
                        ctypes.c_uint64(st.p_reg._buf),
                        ctypes.c_uint64(vmem_buf._buf),
                        ctypes.c_uint64(shared_buf._buf),
                        ctypes.c_uint64(pvt_buf._buf),
                        ctypes.c_uint64(st.exec_mask_buf._buf),
                        ctypes.c_uint64(tmp_buf._buf),
                        ctypes.c_uint64(tex_buf._buf),
                        ctypes.c_uint64(ibo_buf._buf)
                    ], {}, st.snapshot(), pvt_buf))
        else:
            for st, _, _, images, _ in pool: st.reset(images, wgidconstid, group_id)
        for _, _, _, _, pvt_buf in pool: ctypes.memset(pvt_buf._buf, 0, max(pvt_bytes_per_lane * wave_size, 1))
        waves = [(st, bufs, calls_at) for st, bufs, calls_at, _, _ in pool]
        done = [False] * len(waves)
        for _ in range(10_000_000):
            if all(done): return
            active_waves = [w for wi, w in enumerate(waves) if not done[wi]]
            if any(st.barrier_waiting for st, _, _ in active_waves):
                # A workgroup barrier is only valid when every wave still participating
                # in the workgroup reaches it. If another wave has already ended, do not
                # accidentally release the waiting waves as though that wave had arrived.
                if any(done):
                    raise RuntimeError("workgroup reached a barrier after another wave had already completed")
                if _barrier_resume_ready([st for st, _, _ in active_waves]):
                    for st, _, _ in active_waves:
                        st.barrier_waiting = False
            for wi, (st, bufs, calls_at) in enumerate(waves):
                if done[wi] or st.barrier_waiting: continue
                for _ in range(1_000_000):
                    if not st.lane_mask and st.parked:  # every lane is parked: run on (as no-ops) to the next reconvergence point
                        st.pc = min((k for k in st.parked if k >= st.pc), default=min(st.parked))
                    if st.pc in st.parked: _unpark(st, st.pc)
                    pc = st.pc
                    if not 0 <= pc < n_inst:
                        raise RuntimeError(f"wave {wi} reached PC {pc} outside shader ({n_inst} instructions)")
                    is_branch, runners, is_barrier, is_end, is_pred, inst, run_len = _ensure_compiled(pc)
                    st.pc += run_len
                    if trace: print(f"  exec gid=({group_id[0]},{group_id[1]},{group_id[2]}) w={wi} PC={pc}: {inst!r}", flush=True)
                    if is_branch:
                        st.pc = _execute_cat0_branch(runners, st, pc)
                        continue
                    if (calls:=calls_at.get(pc)) is None:  # the kernel argument arrays only depend on the wave, so build them once
                        calls = calls_at[pc] = [(fxn, (ctypes.c_uint64 * len(gl))(*[bufs[g].value for g in gl])) for fxn, gl in runners]
                    if stats is not None:
                        t0 = time.perf_counter()
                        for fxn, args in calls: fxn(args)
                        stats[0] += time.perf_counter() - t0
                        stats[1] += len(calls)
                    else:
                        for fxn, args in calls: fxn(args)
                    if is_pred: _apply_wave_predication(inst, st)
                    if trace_regs: print("    lane0:", _dump_lane0(st), flush=True)
                    if is_end:
                        done[wi] = True
                        break
                    if is_barrier:
                        st.barrier_waiting = True
                        break
                else: raise RuntimeError("exceeded 1M instructions in single wave, likely infinite loop")
        raise RuntimeError("exceeded 10M total scheduling rounds")
    (group_x, group_y, group_z) = group_dims

    t_start = time.perf_counter()
    for gidz, gidy, gidx in itertools.product(range(group_z), range(group_y), range(group_x)):
        _run_workgroup((gidx, gidy, gidz))
    if stats is not None: print(f"[emu stats] {group_x * group_y * group_z} workgroups: {time.perf_counter() - t_start:.3f}s total, {stats[0]:.3f}s in {stats[1]} kernel calls", flush=True)