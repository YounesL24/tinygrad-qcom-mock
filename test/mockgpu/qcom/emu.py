# ruff: noqa: E501
import ctypes
from dataclasses import dataclass
import functools
import hashlib
import itertools
import math
import array
import pathlib
import re
import struct
import tempfile
from typing import Any

from tinygrad.runtime.autogen import mesa
from tinygrad.codegen import to_program
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.engine.worker import get_worker_pool
from tinygrad.helpers import DEBUG, Context, ceildiv, diskcache_get, diskcache_put
from tinygrad.runtime.autogen import libc
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import dtypes

NUM_R_REG = 48 * 4
# half registers are a separate file, or alias the full ones when MERGEDREGS is set (tinygrad leaves it clear)
R_WORDS = NUM_R_REG * 2
NUM_S_REG = 8 * 4
NUM_C_REG = 512 * 4
NUM_A_REG = 2 * 1
NUM_P_REG = 1 * 4
def lane_stride(nlanes: int) -> int: return nlanes + 16  # slot stride
NUM_IMG_DESC = 32

def _c(val, dtype=dtypes.uint32): return UOp.const(val, dtype)

def _as_float32(value: UOp) -> UOp:
    if value.dtype == dtypes.float32: return value
    return value.bitcast(dtypes.float32) if value.dtype.itemsize == 4 else value.cast(dtypes.float32)

def _as_reg_bits(value: UOp) -> UOp:
    if value.dtype == dtypes.float32: return value.bitcast(dtypes.uint32)
    if value.dtype == dtypes.float16: return value.bitcast(dtypes.uint16).cast(dtypes.uint32)
    return value.cast(dtypes.uint32)

def _as_float16(value: UOp) -> UOp:
    # low 16 bits as binary16, typed float immediates are already values
    if value.dtype == dtypes.float16: return value.cast(dtypes.float32)
    if value.dtype == dtypes.float32: return value.cast(dtypes.float16).cast(dtypes.float32)
    return value.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)

def _qcom_demoted_const_f16(ctx, token: str, lane: UOp) -> UOp:
    # a half read of a constant demotes f32 to f16
    _, reg, dim = _qcom_reg_token(token)
    return ctx.read_c_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane).bitcast(dtypes.float32).cast(dtypes.float16).cast(dtypes.float32)

def _qcom_float_operand(ctx, field, lane: UOp) -> UOp:
    raw, mods = _qcom_read_raw_operand(ctx, field.value, lane, field.modifiers)
    try: kind = _qcom_reg_token(field.value)[0]
    except ValueError: kind = ""
    if kind == "hc": value = _qcom_demoted_const_f16(ctx, field.value, lane)
    else: value = _as_float16(raw) if kind in {"hr", "hs"} else _as_float32(raw)
    return _apply_qcom_operand_modifiers(value, mods)

def _qcom_apply_instruction_flags(inst: "Inst", value: UOp) -> UOp:
    if not inst.fields or "sat" not in inst.fields[0].modifiers: return value
    if value.dtype not in {dtypes.float16, dtypes.float32}: raise NotImplementedError(f"(sat) on {_inst_name(inst)!r} with {value.dtype} result is not implemented")
    return value.maximum(UOp.const(0.0, value.dtype)).minimum(UOp.const(1.0, value.dtype))

def _thread_active(exec_mask: UOp, lane_in_wave: UOp) -> UOp:
    return ((exec_mask >> lane_in_wave.cast(dtypes.uint64)) & UOp.const(1, dtypes.uint64)).ne(UOp.const(0, dtypes.uint64))

class _Ctx:
    r_reg: UOp  # buffers, in kernel argument order
    s_reg: UOp
    c_reg: UOp
    a_reg: UOp
    p_reg: UOp
    vmem: UOp
    shared: UOp
    pvt_mem: UOp
    tmp: UOp
    def __init__(self, wave_size = 64, merged = False):
        self._axis_id = 0
        self.wave_size = wave_size  # 64 lanes per hardware wave
        self.merged = merged
        waves = wave_size // 64
        self.stride = stride = lane_stride(wave_size)
        self._bufs = {
            "r_reg": UOp.param(0, dtypes.uint32, R_WORDS * stride, name="r_reg"), "s_reg": UOp.param(1, dtypes.uint32, NUM_S_REG * waves, name="s_reg"),
            "c_reg": UOp.param(2, dtypes.uint32, NUM_C_REG * waves, name="c_reg"), "a_reg": UOp.param(3, dtypes.uint32, NUM_A_REG * waves, name="a_reg"),
            "p_reg": UOp.param(4, dtypes.uint32, NUM_P_REG * stride, name="p_reg"),
            "vmem": UOp.param(5, dtypes.uint8, 1<<48, name="vmem"),  # host VAs reach 2**47
            "shared": UOp.param(6, dtypes.uint8, 32 * 1024 * waves, name="shared"), "pvt_mem": UOp.param(7, dtypes.uint8, 1 << 30, name="pvt_mem"),
            "tmp": UOp.param(9, dtypes.uint32, 4 * stride, name="tmp")}  # scratch for swz, gat and sct
        self._exec_mask = UOp.param(8, dtypes.uint64, waves, name="exec_mask")  # one word per wave
        self.tex = UOp.param(10, dtypes.uint32, 16 * NUM_IMG_DESC, name="tex")  # A6XX_TEX_CONST descriptors
        self.ibo = UOp.param(11, dtypes.uint32, 16 * NUM_IMG_DESC, name="ibo")
        self.wsbase = UOp.param(12, dtypes.uint32, waves, name="wsbase")  # shared memory offset per wave
        self._w: Any = None
        self._l: Any = None
        for name, buf in self._bufs.items(): setattr(self, name, buf)

    def advance(self, done: UOp) -> None:  # later accesses wait for done
        for name, buf in self._bufs.items(): setattr(self, name, buf.after(done))

    def range(self) -> UOp:
        # lane index as nested wave and lane loops, so wave state is invariant in the inner loop
        self._axis_id += 2
        self._w = UOp.range(self.wave_size // 64, self._axis_id - 1, dtype=dtypes.int)
        self._l = UOp.range(64, self._axis_id, dtype=dtypes.int)
        return self._w * _c(64, dtypes.int) + self._l

    def read_exec_mask(self, thread_id: UOp) -> UOp: return self._exec_mask.index(self._w).load()
    def lane_active(self, thread_id: UOp) -> UOp: return _thread_active(self.read_exec_mask(thread_id), self._l)
    def wave_of(self, thread_id: UOp) -> UOp: return self._w

    def _slot(self, reg_num: UOp, reg_dim: UOp) -> UOp: return reg_num.cast(dtypes.int) * _c(4, dtypes.int) + reg_dim.cast(dtypes.int)
    def _half_word(self, half_slot: UOp, thread_id: UOp) -> UOp:  # half slot h is the low or high half of word h // 2
        return _c(0 if self.merged else NUM_R_REG * self.stride, dtypes.int) + (half_slot >> _c(1, dtypes.int)) * _c(self.stride, dtypes.int) + thread_id.cast(dtypes.int)
    def _half_shift(self, half_slot: UOp) -> UOp: return (half_slot & _c(1, dtypes.int)).cast(dtypes.uint32) * _c(16, dtypes.uint32)
    def _set_half(self, old: UOp, val: UOp, shift: UOp) -> UOp:
        return (old & ~(_c(0xFFFF, dtypes.uint32) << shift)) | ((val.cast(dtypes.uint32) & _c(0xFFFF, dtypes.uint32)) << shift)

    def read_r_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, half=False):
        slot = self._slot(reg_num, reg_dim)
        if half: return (self.r_reg.index(self._half_word(slot, thread_id)).load() >> self._half_shift(slot)) & _c(0xFFFF, dtypes.uint32)
        return self.r_reg.index(slot * _c(self.stride, dtypes.int) + thread_id.cast(dtypes.int)).load()

    # stores blend with the old value instead of branching on the exec mask
    def write_r_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None, half=False):
        buf = self.r_reg.after(after) if after is not None else self.r_reg
        slot = self._slot(reg_num, reg_dim)
        active = _thread_active(exec_mask, self._l)
        if half:
            idx = self._half_word(slot, thread_id)
            old = buf.index(idx).load()
            return buf.index(idx).store(active.where(self._set_half(old, val, self._half_shift(slot)), old))
        idx = slot * _c(self.stride, dtypes.int) + thread_id.cast(dtypes.int)
        return buf.index(idx).store(active.where(val.cast(dtypes.uint32), buf.index(idx).load()))

    # s and c registers are per wave
    def _wave_read(self, buf: UOp, size: int, reg_num: UOp, reg_dim: UOp, thread_id: UOp, half: bool):
        slot = self._slot(reg_num, reg_dim)
        word = buf.index(self.wave_of(thread_id) * _c(size, dtypes.int) + (slot // _c(2, dtypes.int) if half else slot)).load()
        return (word >> self._half_shift(slot)) & _c(0xFFFF, dtypes.uint32) if half else word

    def _wave_write(self, buf: UOp, size: int, reg_num: UOp, reg_dim: UOp, thread_id: UOp, val: UOp, half: bool):
        slot = self._slot(reg_num, reg_dim)
        base = self.wave_of(thread_id) * _c(size, dtypes.int)
        if not half: return buf.index(base + slot).store(val.cast(dtypes.uint32))
        idx = base + slot // _c(2, dtypes.int)
        return buf.index(idx).store(self._set_half(buf.index(idx).load(), val, self._half_shift(slot)))

    def read_s_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, half=False): return self._wave_read(self.s_reg, NUM_S_REG, reg_num, reg_dim, thread_id, half)
    def read_c_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, half=False): return self._wave_read(self.c_reg, NUM_C_REG, reg_num, reg_dim, thread_id, half)
    def write_s_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, val: UOp, after: UOp | None = None, half=False):
        return self._wave_write(self.s_reg.after(after) if after is not None else self.s_reg, NUM_S_REG, reg_num, reg_dim, thread_id, val, half)
    def write_c_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, val: UOp, after: UOp | None = None, half=False):
        return self._wave_write(self.c_reg.after(after) if after is not None else self.c_reg, NUM_C_REG, reg_num, reg_dim, thread_id, val, half)

    def read_a_reg(self, reg_num: UOp, thread_id: UOp):
        return self.a_reg.index(self.wave_of(thread_id) * _c(NUM_A_REG, dtypes.int) + reg_num.cast(dtypes.int)).load()
    def write_a_reg(self, reg_num: UOp, thread_id: UOp, val: UOp, after: UOp | None = None):
        buf = self.a_reg.after(after) if after is not None else self.a_reg
        return buf.index(self.wave_of(thread_id) * _c(NUM_A_REG, dtypes.int) + reg_num.cast(dtypes.int)).store(val.cast(dtypes.uint32))

    # one 0/1 word per lane
    def read_p_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp):
        return self.p_reg.index(self._slot(reg_num, reg_dim) * _c(self.stride, dtypes.int) + thread_id.cast(dtypes.int)).load()
    def write_p_reg(self, reg_num: UOp, reg_dim: UOp, thread_id: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None):
        buf = self.p_reg.after(after) if after is not None else self.p_reg
        idx = self._slot(reg_num, reg_dim) * _c(self.stride, dtypes.int) + thread_id.cast(dtypes.int)
        return buf.index(idx).store(_thread_active(exec_mask, self._l).where(val.ne(_c(0, val.dtype)).cast(dtypes.uint32), buf.index(idx).load()))

def _compile_cat0(inst, ctx):
    name = _inst_name(inst).strip().lower()
    if name in {"nop", "end", "predt", "predf", "prede"}: return UOp.sink()  # handled by the scheduler
    if name in {"jump", "call", "ret", "br", "bany", "ball", "brao", "braa", "brax", "bkt", "kill", "emit", "cut", "chmask", "chsh", "flow_rev", "shpe", "getlast", "getone", "shps", "brac"}:
        raise NotImplementedError(f"Category 0 instruction {name!r} is handled by the wave scheduler or needs unsupported subgroup/geometry semantics")
    raise RuntimeError(f"Unknown category 0 instruction: {name!r}")

def _write_qcom_store(ctx: _Ctx, token: str, lane: UOp, value: UOp, after: UOp | None = None):
    kind, reg, dim = _qcom_reg_token(token)
    r, d = _c(reg, dtypes.int), _c(dim, dtypes.int)
    half = kind in {"hr", "hs", "hc"}
    if half:  # raw 16 bits
        if value.dtype == dtypes.float32: value = value.cast(dtypes.float16)
        value = value.bitcast(dtypes.uint16) if value.dtype == dtypes.float16 else value.cast(dtypes.uint16)
    if kind in {"r", "hr"}: return ctx.write_r_reg(r, d, lane, _as_reg_bits(value), ctx.read_exec_mask(lane), after=after, half=half)
    if kind in {"s", "hs"}: return ctx.write_s_reg(r, d, lane, _as_reg_bits(value), after=after, half=half)
    if kind in {"c", "hc"}: return ctx.write_c_reg(r, d, lane, _as_reg_bits(value), after=after, half=half)
    if kind == "a": return ctx.write_a_reg(r, lane, _as_reg_bits(value), after=after)
    if kind == "p": return ctx.write_p_reg(r, d, lane, value, ctx.read_exec_mask(lane), after=after)
    raise ValueError(f"Unsupported QCOM register kind {kind!r} in {token!r}")

def _write_qcom_value(ctx: _Ctx, token: str, lane: UOp, value: UOp): return UOp.sink(_write_qcom_store(ctx, token, lane, value))

def _qcom_reg_component_offset_token(token: str, delta: int) -> str:
    # step a vector operand by components, w carries into the next register
    kind, reg, dim = _qcom_reg_token(token)
    if kind not in {"r", "hr", "c", "hc", "s", "hs"}: raise NotImplementedError(f"Multi-component accesses require r/c/s registers, got {token!r}")
    slot = reg * 4 + dim + delta
    return f"{kind}{slot // 4}.{'xyzw'[slot % 4]}"

_QCOM_VALUE_TYPES = ("u8_32", "f16", "f32", "u16", "u32", "s16", "s32", "u8")

def _qcom_split_type_pair(suffix: str) -> tuple[str, str] | None:
    for src_type in sorted(_QCOM_VALUE_TYPES, key=len, reverse=True):
        if suffix.startswith(src_type) and suffix[len(src_type):] in _QCOM_VALUE_TYPES: return src_type, suffix[len(src_type):]
    return None

_QCOM_TYPE_DTYPES = {"f16": dtypes.float16, "f32": dtypes.float32, "u8": dtypes.uint8, "u16": dtypes.uint16, "u32": dtypes.uint32, "s16": dtypes.int16, "s32": dtypes.int32}

def _qcom_decode_typed(value: UOp, typ: str) -> UOp:
    if typ == "f16": return _as_float16(value)
    if typ == "f32": return _as_float32(value)
    if typ == "u8_32": return value.cast(dtypes.uint8).cast(dtypes.uint32)
    if typ in {"s16", "s32"}: return value.cast(dtypes.uint16 if typ == "s16" else dtypes.uint32).bitcast(_QCOM_TYPE_DTYPES[typ])
    return value.cast(_QCOM_TYPE_DTYPES[typ])

def _qcom_encode_typed(value: UOp, typ: str) -> UOp:
    return value.cast(dtypes.uint8).cast(dtypes.uint32) if typ == "u8_32" else value.cast(_QCOM_TYPE_DTYPES[typ])

def _qcom_u8_source_extends(src: UOp, src_type: str, dst_type: str) -> UOp:
    # 8 bit cov and mov sources sign extend
    return src.bitcast(dtypes.int8) if src_type == "u8" and dst_type != "u8" else src

def _compile_cat1(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 1 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    op, _, suffix = name.partition(".")
    lane = ctx.range()

    if op in {"swz", "gat", "sct"}:
        if len(inst.fields) != (5 if op == "swz" else 6): raise RuntimeError(f"{op} has the wrong number of operands: {inst!r}")
        if op == "swz": vals, targets = [_qcom_read_operand(ctx, f.value, lane, f.modifiers) for f in inst.fields[3:5]], [f.value for f in inst.fields[1:3]]
        elif op == "gat":  # consecutive dst
            vals = [_qcom_read_operand(ctx, f.value, lane, f.modifiers) for f in inst.fields[2:6]]
            targets = [_qcom_reg_component_offset_token(inst.fields[1].value, i) for i in range(4)]
        else:  # consecutive src
            src = inst.fields[5]
            vals = [_qcom_read_operand(ctx, _qcom_reg_component_offset_token(src.value, i), lane, src.modifiers) for i in range(4)]
            targets = [f.value for f in inst.fields[1:5]]
        # parallel move, park the sources in scratch first so a load cannot move past a store to a shared half word
        parked = _end_lane_range(UOp.sink(*[ctx.tmp.index(_c(i * ctx.stride, dtypes.int) + lane).store(_as_reg_bits(v)) for i, v in enumerate(vals)]))
        ctx.advance(parked)
        lane = ctx.range()
        stores: list[UOp] = []
        for i, dst in enumerate(targets):  # chained, half registers share words
            stores.append(_write_qcom_store(ctx, dst, lane, ctx.tmp.index(_c(i * ctx.stride, dtypes.int) + lane).load(), after=stores[-1] if stores else None))
        return UOp.sink(*stores)

    if len(inst.fields) < 3: raise RuntimeError(f"Category 1 instruction is missing operands: {inst!r}")
    dst, src_field = inst.fields[1].value, inst.fields[2]
    if op in {"movs", "movmsk"}: raise NotImplementedError(f"{op} is not modeled")
    pair = _qcom_split_type_pair(suffix) if op in {"cov", "mov"} else None  # suffix is the source and destination type
    if op == "cov" and pair is None: raise NotImplementedError(f"Unsupported conversion type pair in {name!r}")
    if pair is not None:
        src = _qcom_u8_source_extends(_qcom_read_typed_operand(ctx, src_field, lane, pair[0]), *pair)
        return _write_qcom_value(ctx, dst, lane, _qcom_apply_instruction_flags(inst, _qcom_encode_typed(src, pair[1])))
    if op == "mov": return _write_qcom_value(ctx, dst, lane, _qcom_apply_instruction_flags(inst, _qcom_read_operand(ctx, src_field.value, lane, src_field.modifiers)))
    if op in {"mova", "mova1"}: return _write_qcom_value(ctx, dst, lane, _qcom_read_operand(ctx, src_field.value, lane, src_field.modifiers))
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
    # 48 registers per lane, 48-55 are wave shared
    if kind in {"r", "hr"} and reg >= 48: return "s" if kind == "r" else "hs", reg - 48, dim
    return kind, reg, dim


def _qcom_read_value(ctx: _Ctx, token: str, lane: UOp) -> UOp:
    kind, reg, dim = _qcom_reg_token(token)
    if kind == "r": return ctx.read_r_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane)
    if kind == "hr": return ctx.read_r_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane, half=True)
    if kind in {"s", "hs"}: return ctx.read_s_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane, half=(kind == "hs"))
    if kind == "c": return ctx.read_c_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane)
    # hcN is the full constant cN demoted, integer readers take the low 16 bits
    if kind == "hc": return ctx.read_c_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane) & _c(0xFFFF, dtypes.uint32)
    if kind == "a": return ctx.read_a_reg(_c(reg, dtypes.int), lane)
    if kind == "p": return ctx.read_p_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane)
    raise ValueError(f"Unsupported QCOM register kind {kind!r} in {token!r}")


def _parse_qcom_imm(token: str) -> UOp:
    token = token.strip()
    if token.lower().startswith("h("): token = token[1:]  # half immediate
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
        # only non register tokens are immediates
        if _QCOM_REG_RE.fullmatch(token.strip()) is not None or "[" in token or "]" in token:
            raise
        return _parse_qcom_imm(token), mods


def _qcom_read_operand(ctx: _Ctx, token: str, lane: UOp, modifiers: list[str] | tuple[str, ...] | None = None) -> UOp:
    value, mods = _qcom_read_raw_operand(ctx, token, lane, modifiers)
    return _apply_qcom_operand_modifiers(value, mods)


def _qcom_read_typed_operand(ctx: _Ctx, field, lane: UOp, typ: str) -> UOp:
    raw, mods = _qcom_read_raw_operand(ctx, field.value, lane, field.modifiers)
    if typ == "f16" and _QCOM_REG_RE.fullmatch(field.value.strip()) and _qcom_reg_token(field.value)[0] == "hc":
        return _apply_qcom_operand_modifiers(_qcom_demoted_const_f16(ctx, field.value, lane), mods)
    # signed ops on half sources sign extend from bit 15
    if typ == "s32" and field.value.strip().lower().startswith("h"): return _apply_qcom_operand_modifiers(_qcom_decode_typed(raw, "s16").cast(dtypes.int32), mods)
    return _apply_qcom_operand_modifiers(_qcom_decode_typed(raw, typ), mods)

def _round_nearest_even(x: UOp) -> UOp:
    truncated = UOp(Ops.TRUNC, src=(x,))
    away = (x < UOp.const(0.0, x.dtype)).where(truncated - _c(1.0, x.dtype), truncated + _c(1.0, x.dtype))  # next integer away from zero
    frac = (x - truncated).abs()
    is_odd = (truncated.cast(dtypes.int32) & _c(1, dtypes.int32)).ne(_c(0, dtypes.int32))
    return ((frac > UOp.const(0.5, x.dtype)) | (frac.eq(UOp.const(0.5, x.dtype)) & is_odd)).where(away, truncated)  # ties go to even


def _round_away_from_zero(x: UOp) -> UOp:
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
    if name in {"setrm", "flat.b", "mull.u2", "div2.u", "mul2.u"} or name.startswith(("bary.", "mgen.", "shb", "msad")):
        raise NotImplementedError(f"Category 2 instruction {name!r} is not implemented")
    if len(inst.fields) < 3: raise RuntimeError(f"Category 2 instruction is missing operands: {inst!r}")
    lane = ctx.range()
    dst_token, src_fields = inst.fields[1].value, inst.fields[2:]
    src_vals = [] if name == "absneg.s" else [_qcom_read_operand(ctx, f.value, lane, f.modifiers) for f in src_fields]
    u32 = lambda v: v.cast(dtypes.uint32)  # noqa: E731
    f32 = lambda x: UOp.const(x, dtypes.float32)  # noqa: E731

    if name in {"absneg.f", "absneg.s", "sign.f", "floor.f", "ceil.f", "rndne.f", "rndaz.f", "trunc.f", "not.b", "bfrev.b", "clz.s", "clz.b", "cbits.b"}:
        if len(src_fields) != 1: raise RuntimeError(f"{name} expects one source: {inst!r}")
        if name.endswith(".f"): srcf = _qcom_float_operand(ctx, src_fields[0], lane)
        elif name.endswith(".s"): srcf = _qcom_read_typed_operand(ctx, src_fields[0], lane, "s32")
        else: srcf = src_vals[0]
        # absneg only applies the source modifiers
        if name in {"absneg.f", "absneg.s"}: result = srcf
        elif name == "sign.f": result = (srcf < f32(0.0)).where(f32(-1.0), (srcf > f32(0.0)).where(f32(1.0), f32(0.0)))
        elif name == "floor.f": result = srcf.floor()
        elif name == "ceil.f": result = srcf.ceil()
        elif name == "rndne.f": result = _round_nearest_even(srcf)
        elif name == "rndaz.f": result = _round_away_from_zero(srcf)
        elif name == "trunc.f": result = srcf.trunc()
        elif name == "not.b": result = ~u32(src_vals[0])
        elif name == "bfrev.b": result = _qcom_bitreverse32(src_vals[0])
        elif name == "clz.b":  # -1 for 0
            v = u32(src_vals[0])
            result = v.eq(_c(0)).where(_c(0xFFFFFFFF), _qcom_clz32(v))
        elif name == "clz.s":  # leading sign bits, -1 for 0 and ~0
            v = src_vals[0].cast(dtypes.int32)
            flipped = u32(v ^ (v >> _c(31, dtypes.int32)))
            result = flipped.eq(_c(0)).where(_c(0xFFFFFFFF), _qcom_clz32(flipped))
        else: result = _qcom_popcount32(src_vals[0])
        return _write_qcom_value(ctx, dst_token, lane, _qcom_apply_instruction_flags(inst, result))

    if len(src_fields) != 2: raise RuntimeError(f"{name} expects two source operands: {inst!r}")
    if name.startswith(("cmpv.", "cmps.")):
        parts = name.split(".")
        if len(parts) != 3: raise NotImplementedError(f"Unsupported comparison form {name!r}")
        cmp_kind, cmp_type, condition = parts
        if cmp_type == "f": lhs, rhs = _qcom_float_operand(ctx, src_fields[0], lane), _qcom_float_operand(ctx, src_fields[1], lane)
        elif cmp_type in {"u", "s"}: lhs, rhs = (_qcom_read_typed_operand(ctx, f, lane, cmp_type + "32") for f in src_fields)
        else: raise RuntimeError(f"Unknown comparison type in {name!r}")
        # cmpv is a mask, cmps is 0 or 1
        result = _qcom_cmp(lhs, rhs, condition).where(_c(0xFFFFFFFF if cmp_kind == "cmpv" else 1), _c(0))
        return _write_qcom_value(ctx, dst_token, lane, _qcom_apply_instruction_flags(inst, result))

    a, b = src_vals
    float_a, float_b = _qcom_float_operand(ctx, src_fields[0], lane), _qcom_float_operand(ctx, src_fields[1], lane)
    signed_a, signed_b = _qcom_read_typed_operand(ctx, src_fields[0], lane, "s32"), _qcom_read_typed_operand(ctx, src_fields[1], lane, "s32")
    sh = u32(b) & _c(31)
    if name == "add.f": result = float_a + float_b
    elif name == "add.u": result = u32(a) + u32(b)
    elif name == "add.s": result = signed_a + signed_b
    elif name == "sub.u": result = u32(a) - u32(b)
    elif name == "sub.s": result = signed_a - signed_b
    elif name == "mul.f": result = float_a * float_b
    elif name == "min.f": result = float_a.minimum(float_b)
    elif name == "max.f": result = float_a.maximum(float_b)
    elif name in {"min.u", "max.u", "min.s", "max.s"}:
        lhs, rhs = (u32(a), u32(b)) if name.endswith(".u") else (signed_a, signed_b)
        result = lhs.minimum(rhs) if name.startswith("min") else lhs.maximum(rhs)
    elif name == "mul.u24": result = (u32(a) & _c(0xFFFFFF)) * (u32(b) & _c(0xFFFFFF))
    elif name == "mul.s24": result = _qcom_sign_extend(signed_a, 24) * _qcom_sign_extend(signed_b, 24)
    elif name == "and.b": result = u32(a) & u32(b)
    elif name == "or.b": result = u32(a) | u32(b)
    elif name == "xor.b": result = u32(a) ^ u32(b)
    elif name == "shl.b": result = u32(a) << sh
    elif name == "shr.b": result = u32(a) >> sh
    elif name == "ashr.b": result = signed_a >> sh  # s16 extended for half sources
    elif name == "getbit.b": result = (u32(a) >> sh) & _c(1)
    elif name == "mull.u": result = (u32(a) & _c(0xFFFF)) * (u32(b) & _c(0xFFFF))  # low 16 bits
    else: raise RuntimeError(f"Unsupported category 2 instruction: {name!r}")
    return _write_qcom_value(ctx, dst_token, lane, _qcom_apply_instruction_flags(inst, result))


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
    # private memory is interleaved by dword across lanes
    b = addr_i + _c(i, dtypes.int64)
    if mem_kind == "l": return b + ctx.wsbase.index(ctx.wave_of(lane)).load().cast(dtypes.int64)
    if mem_kind != "p": return b
    return (b >> _c(2, dtypes.int64)) * _c(ctx.wave_size * 4, dtypes.int64) + lane.cast(dtypes.int64) * _c(4, dtypes.int64) + (b & _c(3, dtypes.int64))

def _load_mem_value(ctx: _Ctx, mem_kind: str, addr: UOp, suffix: str, lane: UOp) -> UOp:
    mem = {"g": ctx.vmem, "l": ctx.shared, "p": ctx.pvt_mem}[mem_kind]
    active = ctx.lane_active(lane)
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
    assert raw is not None
    # memory holds bits, reinterpret instead of casting
    if dtype in {dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.float16, dtypes.float32}:
        decoded = raw.bitcast(dtype)
        # narrow signed loads sign extend
        if dtype in {dtypes.int8, dtypes.int16}: return decoded.cast(dtypes.int32)
        return decoded
    return raw.cast(dtype)


def _store_mem_value(ctx: _Ctx, mem_kind: str, addr: UOp, value: UOp, suffix: str, lane: UOp):
    mem = {"g": ctx.vmem, "l": ctx.shared, "p": ctx.pvt_mem}[mem_kind]
    active = ctx.lane_active(lane)
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
    if name in {"madsh.u16", "mad.f16.mul2", "mad.f32.mul2", "mad.f16.div2", "mad.f32.div2"}: raise NotImplementedError(f"Category 3 instruction {name!r} is not implemented")
    lane = ctx.range()
    dst, src_fields = inst.fields[1].value, inst.fields[2:5]
    a, b, c = (_qcom_read_operand(ctx, f.value, lane, f.modifiers) for f in src_fields)
    parts = name.split(".")
    u32 = lambda v: v.cast(dtypes.uint32)  # noqa: E731
    s16 = lambda v: v.cast(dtypes.uint16).bitcast(dtypes.int16).cast(dtypes.int32)  # noqa: E731
    s24 = lambda v: _qcom_sign_extend(v, 24)  # noqa: E731
    def f16(field): return _qcom_read_typed_operand(ctx, field, lane, "f16").cast(dtypes.float32)
    def packed_byte(value, byte, signed=True):
        v = (u32(value) >> _c(byte * 8)) & _c(0xFF)
        return v.cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.int32) if signed else v

    if name == "mad.f16": value = (f16(src_fields[0]) * f16(src_fields[1]) + f16(src_fields[2])).cast(dtypes.float16)
    elif name == "mad.f32": value = _qcom_float_operand(ctx, src_fields[0], lane) * _qcom_float_operand(ctx, src_fields[1], lane) + _qcom_float_operand(ctx, src_fields[2], lane)
    elif name == "madsh.m16": value = (((u32(a) & _c(0xFFFF)) * (u32(b) >> _c(16))) << _c(16)) + u32(c)
    elif name == "mad.u16": value = u32(a.cast(dtypes.uint16)) * u32(b.cast(dtypes.uint16)) + u32(c)
    elif name == "mad.s16": value = s16(a) * s16(b) + c.cast(dtypes.int32)
    elif name == "mad.u24": value = (u32(a) & _c(0xFFFFFF)) * (u32(b) & _c(0xFFFFFF)) + u32(c)
    elif name == "mad.s24": value = s24(a) * s24(b) + c.cast(dtypes.int32)
    elif name in {"sel.b16", "sel.b32"}: value = u32(b).ne(_c(0)).where(u32(a), u32(c))  # middle source is the condition
    elif name in {"sel.s16", "sel.s32", "sel.f16", "sel.f32"}: raise NotImplementedError(f"{name}: operand roles are not documented in Mesa's ISA XML and Mesa never emits it")
    elif name in {"sad.s16", "sad.s32"}:
        aa, bb = (s16(a), s16(b)) if name.endswith("s16") else (a.cast(dtypes.int32), b.cast(dtypes.int32))
        value = (aa - bb).abs() + c.cast(dtypes.int32)
    elif name in {"shrm", "shlm", "shrg", "shlg", "andg"}:
        shift = u32(a) & _c(31)
        if name == "shrm": value = (u32(b) >> shift) & u32(c)
        elif name == "shlm": value = (u32(b) << shift) & u32(c)
        elif name == "shrg": value = (u32(b) >> shift) | u32(c)
        elif name == "shlg": value = (u32(b) << shift) | u32(c)
        else: value = (u32(b) & u32(a)) | u32(c)
    elif parts[0] == "dp2acc":  # low or high byte pair
        signed, first_byte = "mixed" in parts[1:], 2 if "high" in parts[1:] else 0
        value = c.cast(dtypes.int32) + sum((packed_byte(a, first_byte + i, signed) * packed_byte(b, first_byte + i, False) for i in range(2)), _c(0, dtypes.int32))
    elif parts[0] == "dp4acc":  # low form only
        if "high" in parts[1:]: raise NotImplementedError("dp4acc.high needs the 64-bit packed-source register model")
        value = c.cast(dtypes.int32) + sum((packed_byte(a, i, "mixed" in parts[1:]) * packed_byte(b, i, False) for i in range(4)), _c(0, dtypes.int32))
    else: raise RuntimeError(f"Unsupported category 3 instruction: {name!r}")
    return _write_qcom_value(ctx, dst, lane, _qcom_apply_instruction_flags(inst, value))


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
    return _write_qcom_value(ctx, dst, lane, _qcom_apply_instruction_flags(inst, result))


def _image_descriptor(ctx, buf: UOp, index: int) -> dict:
    # descriptors change per dispatch so they are read at run time
    w = [buf.index(_c(index * 16 + k, dtypes.int)).load() for k in range(6)]
    def field(word, mask, shift): return (w[word] & _c(mask, dtypes.uint32)) >> _c(shift, dtypes.uint32)
    base = (w[4] & _c(mesa.A6XX_TEX_CONST_4_BASE_LO__MASK, dtypes.uint32)).cast(dtypes.uint64) | ((w[5] & _c(mesa.A6XX_TEX_CONST_5_BASE_HI__MASK, dtypes.uint32)).cast(dtypes.uint64) << _c(32, dtypes.uint64))
    return {"w0": w[0], "base": base, "fmt": field(0, mesa.A6XX_TEX_CONST_0_FMT__MASK, mesa.A6XX_TEX_CONST_0_FMT__SHIFT),
            "width": field(1, mesa.A6XX_TEX_CONST_1_WIDTH__MASK, mesa.A6XX_TEX_CONST_1_WIDTH__SHIFT).cast(dtypes.int),
            "height": field(1, mesa.A6XX_TEX_CONST_1_HEIGHT__MASK, mesa.A6XX_TEX_CONST_1_HEIGHT__SHIFT).cast(dtypes.int),
            "pitch": field(2, mesa.A6XX_TEX_CONST_2_PITCH__MASK, mesa.A6XX_TEX_CONST_2_PITCH__SHIFT).cast(dtypes.uint64)}


def _texel_addr(desc: dict, x: UOp, y: UOp) -> tuple[UOp, UOp, UOp]:
    # (address, in bounds, is f32) of a texel
    inb = (x >= _c(0, dtypes.int)) & (x < desc["width"]) & (y >= _c(0, dtypes.int)) & (y < desc["height"])
    is32 = desc["fmt"].eq(_c(mesa.FMT6_32_32_32_32_FLOAT, dtypes.uint32))
    bpp = is32.where(_c(16, dtypes.uint64), _c(8, dtypes.uint64))
    return desc["base"] + y.cast(dtypes.uint64) * desc["pitch"] + x.cast(dtypes.uint64) * bpp, inb, is32


def _load_texel(ctx, desc: dict, x: UOp, y: UOp, lane: UOp, swizzle: bool) -> list[UOp]:
    # four f32 components, zero out of bounds
    addr, inb, is32 = _texel_addr(desc, x, y)
    valid = inb & ctx.lane_active(lane)
    comps = []
    for i in range(4):
        raw: UOp | None = None
        for j in range(4):  # f16 only has 2 bytes
            off = is32.where(_c(4 * i + j, dtypes.uint64), _c(2 * i + j, dtypes.uint64))
            byte = ctx.vmem.index((addr + off).cast(dtypes.int64).valid(valid & (is32 | _c(j < 2, dtypes.bool)))).load().cast(dtypes.uint32)
            raw = byte if raw is None else raw | (byte << _c(8 * j, dtypes.uint32))
        assert raw is not None
        comps.append(is32.where(raw.bitcast(dtypes.float32), raw.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)))
    if not swizzle: return comps
    out = []
    for shift, mask in ((mesa.A6XX_TEX_CONST_0_SWIZ_X__SHIFT, mesa.A6XX_TEX_CONST_0_SWIZ_X__MASK), (mesa.A6XX_TEX_CONST_0_SWIZ_Y__SHIFT, mesa.A6XX_TEX_CONST_0_SWIZ_Y__MASK),
                        (mesa.A6XX_TEX_CONST_0_SWIZ_Z__SHIFT, mesa.A6XX_TEX_CONST_0_SWIZ_Z__MASK), (mesa.A6XX_TEX_CONST_0_SWIZ_W__SHIFT, mesa.A6XX_TEX_CONST_0_SWIZ_W__MASK)):
        sel = (desc["w0"] & _c(mask, dtypes.uint32)) >> _c(shift, dtypes.uint32)  # 0-3 is rgba, 4 is zero, 5 is one
        v = sel.eq(_c(4, dtypes.uint32)).where(UOp.const(0.0, dtypes.float32), UOp.const(1.0, dtypes.float32))
        for c in (3, 2, 1, 0): v = sel.eq(_c(c, dtypes.uint32)).where(comps[c], v)
        out.append(v)
    return out


def _store_texel(ctx, desc: dict, x: UOp, y: UOp, values: list[UOp], lane: UOp) -> list[UOp]:
    # out of bounds writes are dropped
    addr, inb, is32 = _texel_addr(desc, x, y)
    valid = inb & ctx.lane_active(lane)
    stores = []
    for j in range(16):
        f32_byte = (values[j // 4].bitcast(dtypes.uint32) >> _c(8 * (j % 4), dtypes.uint32)) & _c(0xFF, dtypes.uint32)
        f16_comp = j // 2
        f16_byte = ((values[f16_comp].cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32) >> _c(8 * (j % 2), dtypes.uint32)) & _c(0xFF, dtypes.uint32)) if j < 8 else _c(0, dtypes.uint32)
        byte = is32.where(f32_byte, f16_byte).cast(dtypes.uint8)
        stores.append(ctx.vmem.index((addr + _c(j, dtypes.uint64)).cast(dtypes.int64).valid(valid & (is32 | _c(j < 8, dtypes.bool)))).store(byte))
    return stores


def _write_texel_result(ctx, lane: UOp, dst: str, comps: list[UOp], wrmask: str, typ: str) -> UOp:
    # one register per bit of the write mask
    if typ not in {"f32", "f16"}: raise NotImplementedError(f"image/texture data type {typ!r} is not implemented")
    stores: list[UOp] = []
    for c in "xyzw":
        if c not in wrmask: continue
        v = comps["xyzw".index(c)]
        stores.append(_write_qcom_store(ctx, _qcom_reg_component_offset_token(dst, len(stores)), lane, v.cast(dtypes.float16) if typ == "f16" else v, after=stores[-1] if stores else None))
    return UOp.sink(*stores)


def _compile_cat5(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 5 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    # only isam is modelled, the rest needs sampler state or quads
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
    # typed 2d image store or load through an ibo descriptor
    name = _inst_name(inst).strip().lower()
    parts = name.split(".")
    if len(parts) != 7 or parts[1:4] != ["b", "typed", "2d"] or parts[4] not in {"f32", "f16"} or parts[5] != "4" or parts[6] != "imm" or len(inst.fields) != 4:
        raise NotImplementedError(f"image instruction {name!r} (only typed 2d f32/f16 x4 with an immediate IBO index is implemented)")
    op, typ = parts[0], parts[4]
    lane = ctx.range()
    reg, coord = inst.fields[1].value, inst.fields[2].value
    if op == "stib" and typ == "f16" and _qcom_reg_token(reg)[0] == "r": reg = "h" + reg  # the disassembler omits the h
    x = _qcom_read_operand(ctx, coord, lane).cast(dtypes.uint32).bitcast(dtypes.int32)
    y = _qcom_read_operand(ctx, _qcom_reg_component_offset_token(coord, 1), lane).cast(dtypes.uint32).bitcast(dtypes.int32)
    desc = _image_descriptor(ctx, ctx.ibo, int(inst.fields[3].value, 0))
    if op == "ldib": return _write_texel_result(ctx, lane, reg, _load_texel(ctx, desc, x, y, lane, swizzle=False), "xyzw", typ)
    reads = [_qcom_read_operand(ctx, _qcom_reg_component_offset_token(reg, i), lane) for i in range(4)]
    values = [(_as_float16(r) if typ == "f16" else _as_float32(r)) for r in reads]
    return UOp.sink(*_store_texel(ctx, desc, x, y, values, lane))


def _qcom_read_mem_address(ctx, token: str, lane: UOp) -> tuple[str, UOp]:
    # g is a 64 bit address from a register pair plus offset, l and p a 32 bit base plus offset
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
            addr = addr + _c((-imm if sign == "-" else imm) & 0xFFFFFFFFFFFFFFFF, dtypes.uint64)  # wraps
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
    if any(part in {"a", "k", "b"} for part in name.split(".")[1:]): raise NotImplementedError(f"Special Cat6 addressing mode {name!r} is not implemented")
    op, _, suffix = name.partition(".")
    mem_kind = {"ldg": "g", "stg": "g", "ldl": "l", "stl": "l", "ldp": "p", "stp": "p"}.get(op)
    if mem_kind is None: raise NotImplementedError(f"Cat6 instruction {name!r} (atomic/resource/other form) is not implemented")
    dtype_suffix = suffix or "u32"
    dtype = _qcom_mem_dtype(dtype_suffix)
    if dtype.itemsize == 8: raise NotImplementedError(f"Cat6 {name!r} uses a 64-bit value, which is not implemented")
    if len(inst.fields) not in {3, 4}: raise RuntimeError(f"{name} expects address/value operands and an optional component count: {inst!r}")
    count = 1  # component count
    if len(inst.fields) == 4:
        try: count = int(inst.fields[3].value, 0)
        except ValueError as e: raise NotImplementedError(f"{name} component count must be an immediate: {inst.fields[3].value!r}") from e
    if not 1 <= count <= 4: raise NotImplementedError(f"{name} component count {count} is outside the supported range 1..4")

    lane = ctx.range()
    stores: list[UOp] = []
    if op.startswith("ld"):
        dst = inst.fields[1].value
        space, addr = _qcom_read_mem_address(ctx, inst.fields[2].value, lane)
        if space != mem_kind: raise RuntimeError(f"{name} uses {mem_kind}[...] but operand addresses {space}[...]")
        if count > 1 and _qcom_reg_token(dst)[0] not in {"r", "hr"}: raise NotImplementedError(f"Multi-component {name} destination must be an r/hr register, got {dst!r}")
        for i in range(count):
            dst_i = _qcom_reg_component_offset_token(dst, i) if count > 1 else dst
            value = _load_mem_value(ctx, mem_kind, addr + _c(i * dtype.itemsize, addr.dtype), dtype_suffix, lane)
            # half registers share words so the writes are ordered
            stores.append(_write_qcom_store(ctx, dst_i, lane, value, after=stores[-1] if stores else None))
    elif op.startswith("st"):
        space, addr = _qcom_read_mem_address(ctx, inst.fields[1].value, lane)
        if space != mem_kind: raise RuntimeError(f"{name} uses {mem_kind}[...] but operand addresses {space}[...]")
        src_token = inst.fields[2].value
        # the disassembler omits the h on stl and stp sources
        if dtype_suffix.strip().lower() in {"u8", "u16", "s16", "f16"} and _qcom_reg_token(src_token)[0] == "r": src_token = "h" + src_token
        if count > 1 and _qcom_reg_token(src_token)[0] not in {"r", "hr"}: raise NotImplementedError(f"Multi-component {name} source must be an r/hr register, got {src_token!r}")
        for i in range(count):
            src = _qcom_read_operand(ctx, _qcom_reg_component_offset_token(src_token, i) if count > 1 else src_token, lane, inst.fields[2].modifiers)
            stores.extend(_store_mem_value(ctx, mem_kind, addr + _c(i * dtype.itemsize, addr.dtype), src, dtype_suffix, lane))
    else: raise RuntimeError(f"Unsupported category 6 instruction: {name!r}")
    return UOp.sink(*stores)


def _compile_cat7(inst, ctx):
    name = _inst_name(inst).strip().lower()
    op = name.split(".", 1)[0]
    if op in {"bar", "fence", "icinv", "dccln", "dcinv", "dcflu", "ccinv", "sleep", "nop"}: return UOp.sink()  # lockstep and coherent
    if op in {"lock", "unlock"}: raise NotImplementedError(f"{name} needs shared atomic/lock semantics")
    raise RuntimeError(f"Unsupported category 7 instruction: {name!r}")


def _expand_repeat(inst: "Inst") -> "list[Inst]":
    # (rptN) runs the instruction N+1 times, stepping the destination and (r) sources by a component
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
    compile_cat = [_compile_cat0, _compile_cat1, _compile_cat2, _compile_cat3, _compile_cat4, _compile_cat5, _compile_cat6, _compile_cat7]
    if not 0 <= inst.cat < len(compile_cat): raise RuntimeError(f"Invalid instruction category: {inst.cat}")
    return compile_cat[inst.cat](inst, ctx)

def _end_lane_range(sink):
    # close the lane loops opened by ctx.range()
    rngs = list(sink.ranges)  # still open
    assert len(rngs) in {0, 2}, f"expected the wave and lane loops of one instruction, got {len(rngs)} ranges"
    return UOp.sink(sink.end(*rngs)) if rngs else sink

# programs are cached on disk by instructions and the source from NUM_R_REG to WaveState
_src = pathlib.Path(__file__).read_text()
_EMU_SRC_HASH = hashlib.sha256(_src[_src.index("NUM_R_REG"):_src.index("class WaveState")].encode()).hexdigest()

MAX_BLOCK = 48  # longest fused run

def _block_key(insts: tuple, merged: bool, nlanes: int) -> dict:
    return {"emu": _EMU_SRC_HASH, "insts": hashlib.sha256(repr(insts).encode()).hexdigest(), "merged": int(merged), "nlanes": nlanes}

def _lower_block(insts: tuple, merged: bool, nlanes: int, renderer=None):
    # lower a run of instructions to one CPU program, False if there is nothing to run
    with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
        ctx = _Ctx(nlanes, merged)
        sinks: list[UOp] = []
        for inst in insts:
            sink = _end_lane_range(_compile_inst(inst, ctx))
            if sink.op is Ops.SINK and not sink.src: continue
            sinks.append(sink)
            ctx.advance(sink)
        if not sinks: return False
        prg = to_program(UOp.sink(*sinks).replace(arg=KernelInfo(name=f"qcom_{_inst_name(insts[0])}_x{len(insts)}")), renderer or Device['CPU'].renderer)
        return prg.to_elf(), list(prg.arg.globals)  # binary and buffer order

_lower_block_task = lambda args: (args[0], _lower_block(*args[1:]))  # noqa: E731

_PROGRAMS: dict[tuple[str, bool, int], Any] = {}

def _load_program(insts: tuple, merged: bool, nlanes: int, lower: bool = True):
    key = _block_key(insts, merged, nlanes)
    if (prg:=_PROGRAMS.get((key["insts"], merged, nlanes))) is None and (prg:=diskcache_get("qcom_emu_elf", key)) is None and lower:
        prg = diskcache_put("qcom_emu_elf", key, _lower_block(insts, merged, nlanes))
    if prg is not None: _PROGRAMS[(key["insts"], merged, nlanes)] = prg
    return prg

@functools.cache
def _get_block_runner(insts: tuple, merged: bool, nlanes: int):
    prg = _load_program(insts, merged, nlanes)
    return (prg[1], Device['CPU'].runtime(prg[0])) if prg is not False else None

def _precompile(blocks: list[tuple], merged: bool, nlanes: int) -> None:
    # lower uncached blocks in the worker pool
    todo = {k["insts"]: b for b in blocks if _load_program(b, merged, nlanes, lower=False) is None and (k:=_block_key(b, merged, nlanes))}
    if len(todo) < 6 or (pool:=get_worker_pool()) is None: return  # too few to be worth it
    renderer, keys = Device['CPU'].renderer, list(todo)
    try:
        for i, prg in pool.imap_unordered(_lower_block_task, [(i, todo[k], merged, nlanes, renderer) for i, k in enumerate(keys)]):
            diskcache_put("qcom_emu_elf", _block_key(todo[keys[i]], merged, nlanes), prg)
            _PROGRAMS[(keys[i], merged, nlanes)] = prg
    except Exception: pass  # reported if the block runs

def _compile_run(pc, decoded_shader, n, merged, nlanes):
    # compile [pc, pc+n) as one kernel
    insts = tuple(i for inst in decoded_shader[pc:pc+n] for i in _expand_repeat(inst))
    try: return _get_block_runner(insts, merged, nlanes)
    except Exception:
        for inst in insts:
            try: _end_lane_range(_compile_inst(inst, _Ctx(nlanes, merged)))
            except Exception as e: raise RuntimeError(f"[emu] Failed to compile {inst!r}: {type(e).__name__}: {e}") from e
        raise


class WaveState:
    # up to 16 hardware waves run as one wide wave, lane L is in wave L // 64
    def __init__(self, wave_size, shader_constants, wgidconstid, wgsizeconstid, localidregid, wgids, wgsize, local_dims):
        self.wave_size = wave_size
        self.n_waves = n_waves = wave_size // 64
        threads = math.prod(local_dims)
        self.waves_per_group = wpg = ceildiv(threads, 64)
        assert n_waves % wpg == 0 and len(wgids) <= n_waves // wpg
        # existing lanes per workgroup slot
        self.group_masks = [sum(((1 << min(64, max(0, threads - k * 64))) - 1) << (64 * (g * wpg + k)) for k in range(wpg)) for g in range(n_waves // wpg)]
        self.launch_mask = functools.reduce(lambda a, b: a | b, self.group_masks[:len(wgids)], 0)
        self.lane_mask = self.exec_mask = self.launch_mask  # running lanes
        self.parked: dict[int, int] = {}  # lanes parked at a pc, rejoined when the wave gets there
        self.exec_mask_buf = Buffer('CPU', 8 * n_waves).ensure_allocated()
        self._push_exec_mask()
        self.predication_mask, self.predication_mode, self.call_stack, self.pc = 0, "none", [], 0

        self.stride = lane_stride(wave_size)
        self.r_reg = Buffer('CPU', R_WORDS * self.stride * 4)  # r0 - r47
        self.s_reg = Buffer('CPU', NUM_S_REG * 4 * n_waves)  # r48 - r55
        self.c_reg = Buffer('CPU', NUM_C_REG * 4 * n_waves)
        self.a_reg = Buffer('CPU', NUM_A_REG * 4 * n_waves)
        self.p_reg = Buffer('CPU', NUM_P_REG * self.stride * 4)  # 0 or 1 per lane
        self._r_reg_mv, self._s_reg_mv, self._c_reg_mv = (b.host.view(fmt='I').mv for b in (self.r_reg, self.s_reg, self.c_reg))
        for b in (self.r_reg, self.s_reg, self.c_reg, self.a_reg, self.p_reg): ctypes.memset(b._buf, 0, b.nbytes)

        consts = [int(v) & 0xFFFFFFFF for v in shader_constants[:NUM_C_REG]]
        c_image = struct.pack(f"{len(consts)}I", *consts)
        for w in range(n_waves):
            ctypes.memmove(self.c_reg._buf + w * NUM_C_REG * 4, c_image, len(c_image))
            # regid is register << 2 | component, 0xFC is unused
            if wgsizeconstid != 0xFC:
                for i in range(3): self._write_c_reg((wgsizeconstid + i) >> 2, (wgsizeconstid + i) & 3, wgsize[i], wave=w)
            if localidregid != 0xFC and w // wpg < len(wgids):
                positions = range((w % wpg) * 64, (w % wpg) * 64 + min(64, max(0, threads - (w % wpg) * 64)))
                for j, ids in enumerate(([p % local_dims[0] for p in positions], [(p // local_dims[0]) % local_dims[1] for p in positions],
                                         [p // (local_dims[0] * local_dims[1]) for p in positions])):
                    slot = ((localidregid + j) >> 2) * 4 + ((localidregid + j) & 3)
                    self._r_reg_mv[slot * self.stride + w * 64:slot * self.stride + w * 64 + len(ids)] = array.array("I", ids)
        for g, wgid in enumerate(wgids): self._write_wgid(wgidconstid, wgid, g)

    def _push_exec_mask(self):
        ctypes.memmove(self.exec_mask_buf._buf, (self.exec_mask & ((1 << self.wave_size) - 1)).to_bytes(8 * self.n_waves, "little"), 8 * self.n_waves)

    def _write_s_reg(self, reg_num, reg_dim, val, wave=0): self._s_reg_mv[wave * NUM_S_REG + reg_num * 4 + reg_dim] = val
    def _write_c_reg(self, reg_num, reg_dim, val, wave=0): self._c_reg_mv[wave * NUM_C_REG + reg_num * 4 + reg_dim] = val
    def _read_p_reg(self, reg_num, reg_dim):
        # read the low bytes as binary digits
        raw = ctypes.string_at(self.p_reg._buf + (reg_num * 4 + reg_dim) * self.stride * 4, self.wave_size * 4)
        return int(raw[::4].translate(_BIT_DIGITS)[::-1], 2)

    def _write_wgid(self, wgidconstid, wgid, group):
        # workgroup ids live in r48.x, y and z
        if wgidconstid == 0xFC: return
        for w in range(group * self.waves_per_group, (group + 1) * self.waves_per_group):
            for i in range(3):
                reg = (wgidconstid + i) >> 2
                if reg >= 48: self._write_s_reg(reg - 48, (wgidconstid + i) & 3, wgid[i], wave=w)
                else: self._write_c_reg(reg, (wgidconstid + i) & 3, wgid[i], wave=w)

    def snapshot(self) -> tuple[bytes, bytes, bytes]:  # fresh batch state
        return tuple(ctypes.string_at(b._buf, b.nbytes) for b in (self.r_reg, self.c_reg, self.s_reg))  # type: ignore[return-value]

    def reset(self, images: tuple[bytes, bytes, bytes], wgidconstid, wgids) -> None:
        for buf, img in zip((self.r_reg, self.c_reg, self.s_reg), images): ctypes.memmove(buf._buf, img, len(img))
        for b in (self.a_reg, self.p_reg): ctypes.memset(b._buf, 0, b.nbytes)
        for g, wgid in enumerate(wgids): self._write_wgid(wgidconstid, wgid, g)
        self.launch_mask = functools.reduce(lambda a, b: a | b, self.group_masks[:len(wgids)], 0)
        self.lane_mask = self.exec_mask = self.launch_mask
        self._push_exec_mask()
        self.parked = {}
        self.predication_mask, self.predication_mode, self.call_stack, self.pc = 0, "none", [], 0


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

def _inst_name(inst: Inst) -> str: return inst.fields[0].value if inst.fields else ""
def _is_end_inst(inst: Inst) -> bool: return inst.cat == 0 and _inst_name(inst).strip().lower() == "end"
def _is_barrier_inst(inst: Inst) -> bool: return inst.cat == 7 and _inst_name(inst).strip().lower().split(".", 1)[0] == "bar"
def _is_predication_enable_inst(inst: Inst) -> bool: return inst.cat == 0 and _inst_name(inst) in {"predt", "predf"}
def _is_predication_disable_inst(inst: Inst) -> bool: return inst.cat == 0 and _inst_name(inst) == "prede"

_QCOM_TEX_TYPES = {"f32", "f16", "u32", "u16", "s32", "s16", "u8"}  # cat5 data types
# float lookup immediates that the disassembler prints by name
_QCOM_FLUT_IMMS = {"e": math.e, "pi": math.pi, "1/pi": 1/math.pi, "1/log2(e)": 1/math.log2(math.e), "log2(e)": math.log2(math.e),
                   "1/log2(10)": 1/math.log2(10), "log2(10)": math.log2(10)}
_QCOM_FLOAT_IMM_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+e[+-]?\d+|inf|nan)", re.IGNORECASE)

def _split_qcom_modifiers(token: str) -> tuple[str, list[str]]:
    # (value, modifiers), header flags like (jp) and (sy) are dropped
    expr, mods = token.strip(), list[str]()
    while (match:=re.match(r"^\(([^()]+)\)", expr)) is not None:
        mod = match.group(1).strip().lower()
        # float immediates are printed in parentheses
        if (_QCOM_FLOAT_IMM_RE.fullmatch(mod) or mod in _QCOM_FLUT_IMMS) and match.end() == len(expr): return mod, mods
        if mod in {"neg", "n", "abs", "a", "bnot", "not", "sat", "r"} or re.fullmatch(r"rpt\d+", mod) or mod in _QCOM_TEX_TYPES or re.fullmatch(r"[xyzw]{1,4}", mod):
            mods.append(mod)
        expr = expr[match.end():].strip()
    return expr, mods

def _apply_qcom_operand_modifiers(value: UOp, modifiers: list[str] | tuple[str, ...] | None) -> UOp:
    unsigned = value.dtype in {dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64}
    for mod in modifiers or []:
        if mod == "abs" or mod == "a": value = value.abs()  # no-op on unsigned views
        elif mod in {"neg", "n"}: value = (~value + _c(1, value.dtype) if unsigned else -value).cast(value.dtype)
        elif mod in {"bnot", "not"}: value = (~value if value.dtype in dtypes.ints else ~value.cast(dtypes.uint32)).cast(value.dtype)
    return value

def _parse_line(line):
    fields: list[Field] = []
    pending: list[str] = []
    for word in line.replace(",", "").split():
        value, modifiers = _split_qcom_modifiers(word)
        # a word of only flags keeps its modifiers for the next field
        if not value:
            pending += modifiers
            continue
        fields.append(Field(value, tuple(pending + modifiers)))
        pending = []
    return fields

def _parse_instructions(disassembled_shader):
    # labels refer to instruction indexes
    instructions: list[list[Field]] = []
    jump_locations: dict[str, int] = {}
    for raw_line in disassembled_shader.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"): continue
        if line.endswith(":"):
            label = line[:-1].strip()
            if label in jump_locations: raise RuntimeError(f"duplicate IR3 branch label {label!r}")
            jump_locations[label] = len(instructions)
        elif fields:=_parse_line(line): instructions.append(fields)
    return instructions, jump_locations

def _decode_program(shader):
    # two words per instruction
    if len(shader) % 2: raise ValueError(f"IR3 shader has an odd number of 32-bit words: {len(shader)}")
    shader_buffer = bytearray(array.array("I", [int(w) & 0xFFFFFFFF for w in shader]).tobytes())
    parsed_list, jump_locations = _parse_instructions(_disassemble_shader((ctypes.c_char * len(shader_buffer)).from_buffer(shader_buffer)))
    if len(parsed_list) != len(shader) // 2: raise RuntimeError(f"IR3 disassembler produced {len(parsed_list)} instructions for {len(shader) // 2} encoded instructions")
    # the category is bits 61..63
    return [Inst(int(shader[2*i + 1]) >> 29 & 0x7, tuple(f)) for i, f in enumerate(parsed_list)], jump_locations

_ANALYSIS: dict[tuple, tuple] = {}

def _run_len(pc, special, leaders):
    # a run stops at branch targets and scheduler instructions
    n = 1
    while pc + n < len(special) and n < MAX_BLOCK and not special[pc + n] and (pc + n) not in leaders: n += 1
    return n

def _analyse(shader, mergedregs: bool, nlanes: int) -> tuple:
    # per shader state shared by every dispatch
    key = (hashlib.sha256(array.array("I", [int(w) & 0xFFFFFFFF for w in shader]).tobytes()).digest(), mergedregs, nlanes)
    if key in _ANALYSIS: return _ANALYSIS[key]
    decoded_shader, jump_locations = _decode_program(shader)
    n_inst = len(decoded_shader)
    special = [_is_cat0_branch(i) or _is_predication_enable_inst(i) or _is_predication_disable_inst(i) or _is_barrier_inst(i) or _is_end_inst(i) for i in decoded_shader]
    leaders = {0} | {_branch_target(i.fields[-1].value, pc, jump_locations, n_inst) for pc, i in enumerate(decoded_shader) if _is_cat0_branch(i) and i.fields and _inst_name(i) != "ret"}
    blocks, pc = [], 0
    while pc < n_inst:
        n = 1 if special[pc] else _run_len(pc, special, leaders)
        if not special[pc]: blocks.append(tuple(i for inst in decoded_shader[pc:pc + n] for i in _expand_repeat(inst)))
        pc += n
    _precompile(blocks, mergedregs, nlanes)
    _ANALYSIS[key] = (decoded_shader, jump_locations, special, leaders, {})
    return _ANALYSIS[key]

def _refresh_exec_mask(state: WaveState) -> None:
    if state.predication_mode == "none": state.exec_mask = state.lane_mask
    else: state.exec_mask = state.lane_mask & (state.predication_mask if state.predication_mode == "true" else ~state.predication_mask)
    state._push_exec_mask()

def _apply_wave_predication(inst: Inst, state: WaveState) -> None:
    name = _inst_name(inst).lower()
    if name in {"predt", "predf"}:  # snapshots p0.x
        state.predication_mask = state._read_p_reg(0, 0) & ((1 << state.wave_size) - 1)
        state.predication_mode = "true" if name == "predt" else "false"
    else: state.predication_mode = "none"
    _refresh_exec_mask(state)

def _is_cat0_branch(inst: Inst) -> bool:
    return inst.cat == 0 and _inst_name(inst).strip().lower() in {"jump", "call", "ret", "br", "bany", "ball", "brao", "braa"}

def _branch_target(token: str, pc: int, jump_locations: dict[str, int], shader_len: int) -> int:
    token = token.strip().lstrip("#")
    label = re.fullmatch(r"(?:l|fxn)(\d+)", token, re.IGNORECASE)
    if token in jump_locations: target = jump_locations[token]  # l0 is not necessarily instruction 0
    elif label is not None: target = int(label.group(1))
    else:  # signed relative displacement
        try: offset = int(token, 0)
        except ValueError as e: raise NotImplementedError(f"Cannot resolve IR3 branch target {token!r}; known labels: {tuple(jump_locations)}") from e
        if 0 <= offset <= 0xFFFFFFFF and offset & 0x80000000: offset -= 1 << 32
        target = pc + offset
    if not 0 <= target < shader_len: raise RuntimeError(f"IR3 branch from PC {pc} resolves to out-of-range target {target} (shader has {shader_len} instructions)")
    return target

def _branch_info(inst: Inst, pc: int, jump_locations: dict[str, int], shader_len: int) -> tuple:
    # (name, target, ((invert, p register, component), ...))
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
    name, target, preds = info
    if name == "ret":
        if not state.call_stack: raise RuntimeError(f"IR3 ret at PC {pc} with an empty call stack")
        return state.call_stack.pop()
    if name == "call":
        state.call_stack.append(pc + 1)
        return target
    if name == "jump": return _goto(state, pc, target)

    # a6xx divergence, lanes taking a forward branch park at the target and rejoin when the wave gets there, lanes leaving a loop park after it
    masks = [(state._read_p_reg(reg, comp) ^ (state.launch_mask if invert else 0)) & state.launch_mask for invert, reg, comp in preds]
    active = state.exec_mask & state.lane_mask
    if name == "br": taken = masks[0] & active
    elif name in {"bany", "ball"}:  # reduce over each 64 lane wave
        taken = 0
        for w in range(state.n_waves):
            aw, mw = (active >> (64 * w)) & 0xFFFFFFFFFFFFFFFF, (masks[0] >> (64 * w)) & 0xFFFFFFFFFFFFFFFF
            if (mw & aw if name == "bany" else aw and (mw & aw) == aw): taken |= aw << (64 * w)
    else: taken = ((masks[0] | masks[1]) if name == "brao" else (masks[0] & masks[1])) & active
    if taken == active: return _goto(state, pc, target)
    if not taken: return pc + 1
    if target > pc: _park(state, target, taken)  # taken lanes wait at the target
    else: _park(state, pc + 1, active & ~taken)  # lanes leaving the loop wait after it
    return pc + 1 if target > pc else target

def _park(state: WaveState, pc: int, lanes: int) -> None:
    state.parked[pc] = state.parked.get(pc, 0) | lanes
    state.lane_mask &= ~lanes
    _refresh_exec_mask(state)

def _goto(state: WaveState, pc: int, target: int) -> int:
    # a forward jump cannot skip parked lanes, park and fall through
    if target > pc and any(pc < k < target for k in state.parked):
        _park(state, target, state.lane_mask)
        return pc + 1
    return target

def _unpark(state: WaveState, pc: int) -> None:
    state.lane_mask |= state.parked.pop(pc)
    _refresh_exec_mask(state)


def _descriptor_buffer(words, what: str) -> Buffer:
    # only linear 2D RGBA32F and RGBA16F is modelled, refuse the rest
    words = list(words)[:16 * NUM_IMG_DESC]
    buf = Buffer('CPU', 16 * NUM_IMG_DESC * 4).ensure_allocated()
    ctypes.memset(buf._buf, 0, 16 * NUM_IMG_DESC * 4)
    for i, w in enumerate(words): ctypes.c_uint32.from_address(buf._buf + 4 * i).value = int(w)
    for d in range(len(words) // 16):
        d0, d2, d4 = words[16 * d], words[16 * d + 2], words[16 * d + 4]
        if not d4 and not words[16 * d + 5]: continue  # unused
        fmt = (d0 & mesa.A6XX_TEX_CONST_0_FMT__MASK) >> mesa.A6XX_TEX_CONST_0_FMT__SHIFT
        if (d0 & mesa.A6XX_TEX_CONST_0_TILE_MODE__MASK) != 0 or (d2 & mesa.A6XX_TEX_CONST_2_TYPE__MASK) >> mesa.A6XX_TEX_CONST_2_TYPE__SHIFT != mesa.A6XX_TEX_2D or \
           fmt not in {mesa.FMT6_32_32_32_32_FLOAT, mesa.FMT6_16_16_16_16_FLOAT}:
            raise NotImplementedError(f"{what} descriptor {d} is not a linear 2D RGBA32F/RGBA16F image: {[hex(int(w)) for w in words[16 * d:16 * d + 6]]}")
    return buf

WAVES_PER_CALL = 16  # waves per wide wave

def run_asm(
    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
    global_dims, group_dims, local_dims, kerneldim, wave_size,
    shared_memsize, pvt_memsize_peritem, pvt_mem_base,
    wgidconstid, wgsizeconstid, wgoffsetconstid, localidregid, linearlocalidregid, mergedregs=False
):
    if wave_size != 64: raise NotImplementedError(f"only wave64 is supported, got {wave_size}")
    waves_per_group = ceildiv(math.prod(local_dims), wave_size)
    total_groups = math.prod(group_dims)
    groups_per_batch = min(max(1, WAVES_PER_CALL // waves_per_group), total_groups)
    n_waves = groups_per_batch * waves_per_group
    nlanes = n_waves * wave_size
    shared_stride = 32 * 1024  # tinygrad writes bytes into SHARED_SIZE so every workgroup gets the maximum
    pvt_bytes = max(pvt_memsize_peritem * 512 * nlanes, 1)  # 512 byte units

    decoded_shader, jump_locations, special, leaders, program = _analyse(shader, mergedregs, nlanes)
    n_inst = len(decoded_shader)
    trace = DEBUG >= 5  # ContextVar reads are slow

    def _ensure_compiled(pc):
        # worked out once per pc
        if pc not in program:
            inst = decoded_shader[pc]
            if _is_cat0_branch(inst): program[pc] = (True, _branch_info(inst, pc, jump_locations, n_inst), False, False, False, inst, 1)
            else:
                n = 1 if special[pc] else _run_len(pc, special, leaders)
                runner = _compile_run(pc, decoded_shader, n, mergedregs, nlanes)
                program[pc] = (False, [(runner[1].fxn, runner[0])] if runner is not None else [], _is_barrier_inst(inst), _is_end_inst(inst),
                               _is_predication_enable_inst(inst) or _is_predication_disable_inst(inst), inst, n)
                if DEBUG >= 3: print(f"[emu] PC={pc}..{pc + n - 1}: {inst!r}")
        return program[pc]

    all_groups = list(itertools.product(range(group_dims[2]), range(group_dims[1]), range(group_dims[0])))
    batches = [[(gx, gy, gz) for gz, gy, gx in all_groups[i:i + groups_per_batch]] for i in range(0, total_groups, groups_per_batch)]
    shared_buf = Buffer('CPU', shared_stride * groups_per_batch).ensure_allocated()
    pvt_buf = Buffer('CPU', pvt_bytes).ensure_allocated()
    tmp_buf = Buffer('CPU', 4 * lane_stride(nlanes) * 4).ensure_allocated()
    tex_buf, ibo_buf = _descriptor_buffer(shader_tex_constants, "texture"), _descriptor_buffer(shader_uav, "IBO")
    wsbase_buf = Buffer('CPU', 4 * n_waves).ensure_allocated()
    for w in range(n_waves): ctypes.c_uint32.from_address(wsbase_buf._buf + 4 * w).value = (w // waves_per_group) * shared_stride
    vmem_buf = Buffer("CPU", 1<<48, options=BufferSpec(external_ptr=0)).ensure_allocated()
    st = WaveState(nlanes, shader_constants, wgidconstid, wgsizeconstid, localidregid, batches[0], local_dims, local_dims)
    snapshot = st.snapshot()
    calls_at: dict[int, list] = {}
    bufs = [ctypes.c_uint64(b._buf) for b in (st.r_reg, st.s_reg, st.c_reg, st.a_reg, st.p_reg, vmem_buf, shared_buf, pvt_buf, st.exec_mask_buf, tmp_buf,
                                                tex_buf, ibo_buf, wsbase_buf)]

    for i, group_ids in enumerate(batches):
        if i: st.reset(snapshot, wgidconstid, group_ids)
        ctypes.memset(shared_buf._buf, 0, shared_stride * groups_per_batch)
        ctypes.memset(pvt_buf._buf, 0, pvt_bytes)
        for _ in range(10_000_000):
            if not st.lane_mask and st.parked:  # run on to the next reconvergence point
                st.pc = min((k for k in st.parked if k >= st.pc), default=min(st.parked))
            if st.pc in st.parked: _unpark(st, st.pc)
            pc = st.pc
            if not 0 <= pc < n_inst: raise RuntimeError(f"reached PC {pc} outside shader ({n_inst} instructions)")
            is_branch, runners, is_barrier, is_end, is_pred, inst, run_len = _ensure_compiled(pc)
            st.pc += run_len
            if trace: print(f"  exec gids={group_ids[0]}..{group_ids[-1]} PC={pc}: {inst!r}", flush=True)
            if is_branch:
                st.pc = _execute_cat0_branch(runners, st, pc)
                continue
            if (calls:=calls_at.get(pc)) is None:  # the kernel argument arrays do not change, so build them once
                calls = calls_at[pc] = [(fxn, (ctypes.c_uint64 * len(gl))(*[bufs[g].value for g in gl])) for fxn, gl in runners]
            for fxn, args in calls: fxn(args)
            if is_pred: _apply_wave_predication(inst, st)
            # a barrier needs no rendezvous: the waves of a workgroup are lanes of the same wide wave, so they are in step
            if is_end:  # the running lanes end; lanes parked elsewhere still have to get here
                st.lane_mask = 0
                _refresh_exec_mask(st)
                if not st.parked: break
        else: raise RuntimeError("exceeded 10M scheduling steps in one batch, likely an infinite loop")
