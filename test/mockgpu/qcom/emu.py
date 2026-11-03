# ruff: noqa: E501
import ctypes
from dataclasses import dataclass
import functools
import itertools
import math
import re
import tempfile
from typing import Any, Callable

from tinygrad.runtime.autogen import mesa
from tinygrad.codegen import to_program
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.engine.realize import get_runtime
from tinygrad.helpers import DEBUG, Context, colored
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


def _qcom_float_operand(ctx, field, lane: UOp) -> UOp:
    """Read an operand as a float before applying its typed abs/neg modifiers."""
    raw, mods = _qcom_read_raw_operand(ctx, field.value, lane, field.modifiers)
    try:
        kind, _, _ = _qcom_reg_token(field.value)
    except ValueError:
        kind = ""
    value = _as_float16(raw) if kind in {"hr", "hs", "hc"} else _as_float32(raw)
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
    __slots__ = ("_axis_id", "wave_size", "r_reg", "p_reg", "merged")
    # Half registers are packed into the low/high halves of the full-width register buffer.
    s_reg = UOp.param(1, dtypes.uint32, NUM_S_REG, name="s_reg")
    c_reg = UOp.param(2, dtypes.uint32, NUM_C_REG, name="c_reg")
    a_reg = UOp.param(3, dtypes.uint32, NUM_A_REG, name="a_reg")
    vmem = UOp.param(5, dtypes.uint8, 1<<48, name="vmem")  # host VAs reach 2**47
    shared = UOp.param(6, dtypes.uint8, 32 * 1024, name="shared")
    pvt_mem = UOp.param(7, dtypes.uint8, 1 << 30, name="pvt_mem")
    _exec_mask = UOp.param(8, dtypes.uint64, 1, name="exec_mask")

    _r_reg_cache: dict[int, UOp] = {}

    def __init__(self, wave_size = 64, merged = False):
        self._axis_id = 0
        self.wave_size = wave_size
        self.merged = merged
        if wave_size not in _Ctx._r_reg_cache: _Ctx._r_reg_cache[wave_size] = UOp.param(0, dtypes.uint32, R_WORDS * wave_size, name="r_reg")
        self.r_reg = _Ctx._r_reg_cache[wave_size]
        self.p_reg = UOp.param(4, dtypes.uint32, NUM_P_REG * wave_size, name="p_reg")

    def range(self, n: int | None = None) -> UOp:
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
    if kind not in {"r", "hr"}:
        raise NotImplementedError(f"Multi-component Cat6 accesses require r/hr registers, got {token!r}")
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
        # these move in parallel (swz can swap two registers): no store may run before every source has been read
        reads = UOp.group(*vals)
        return UOp.sink(*[_write_qcom_store(ctx, dst, lane, val, after=reads) for dst, val in zip(targets, vals)])

    if len(inst.fields) < 3: raise RuntimeError(f"Category 1 instruction is missing operands: {inst!r}")
    dst_field, src_field = inst.fields[1], inst.fields[2]
    if op == "cov":
        if "." not in name: raise RuntimeError(f"Conversion instruction lacks type pair: {name!r}")
        pair = _qcom_split_type_pair(name.split(".", 1)[1])
        if pair is None: raise NotImplementedError(f"Unsupported conversion type pair in {name!r}")
        src_type, dst_type = pair
        src = _qcom_read_typed_operand(ctx, src_field, lane, src_type)
        result = _qcom_apply_instruction_flags(inst, _qcom_encode_typed(src, dst_type))
        return _write_qcom_value(ctx, dst_field.value, lane, result)

    if op == "mov":
        # The mnemonic carries source and destination types (e.g. mov.f32f32 or
        # mov.u32u32). Decode modifiers in the source type, then encode the destination.
        suffix = name.split(".", 1)[1] if "." in name else ""
        pair = _qcom_split_type_pair(suffix)
        if pair is not None:
            src_type, dst_type = pair
            src = _qcom_read_typed_operand(ctx, src_field, lane, src_type)
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


_QCOM_REG_RE = re.compile(r"(?P<kind>[a-zA-Z]+)(?P<reg>\d+)(?:\.(?P<dim>x|y|z|w))?", re.IGNORECASE)


def _qcom_reg_token(token: str) -> tuple[str, int, int]:
    token, _ = _split_qcom_modifiers(token)
    match = _QCOM_REG_RE.fullmatch(token.strip())
    if match is None: raise ValueError(f"Unsupported QCOM register token: {token!r}")
    kind = match.group("kind").lower()
    reg = int(match.group("reg"))
    dim = {"x": 0, "y": 1, "z": 2, "w": 3}.get(match.group("dim") or "x", 0)
    return kind, reg, dim


def _qcom_read_value(ctx: _Ctx, token: str, lane: UOp) -> UOp:
    kind, reg, dim = _qcom_reg_token(token)
    if kind == "r": return ctx.read_r_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane)
    if kind == "hr": return ctx.read_r_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane, half=True)
    if kind in {"s", "hs"}: return ctx.read_s_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), half=(kind == "hs"))
    if kind in {"c", "hc"}: return ctx.read_c_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), half=(kind == "hc"))
    if kind == "a": return ctx.read_a_reg(_c(reg, dtypes.int))
    if kind == "p": return ctx.read_p_reg(_c(reg, dtypes.int), _c(dim, dtypes.int), lane)
    raise ValueError(f"Unsupported QCOM register kind {kind!r} in {token!r}")


def _parse_qcom_imm(token: str) -> UOp:
    token = token.strip()
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
    elif name == "ashr.b": result = a.cast(dtypes.int32) >> (b.cast(dtypes.uint32) & _c(31, dtypes.uint32))
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
        byte = mem.index((safe_addr + _c(i, dtypes.int64)).valid(active)).load().cast(raw_dtype)
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
        stores.append(mem.index((addr_i + _c(i, dtypes.int64)).valid(active)).store(byte))
    return stores


def _compile_cat3(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 3 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    if len(inst.fields) != 5: raise RuntimeError(f"Category 3 instruction requires destination and three sources: {inst!r}")
    if name in {"madsh.u16", "madsh.m16", "mad.f16.mul2", "mad.f32.mul2", "mad.f16.div2", "mad.f32.div2"}:
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


def _compile_cat5(inst, ctx):
    if not inst.fields: raise RuntimeError(f"Category 5 instruction is missing a name: {inst!r}")
    name = _inst_name(inst).strip().lower()
    # Cat5 is not a homogeneous ALU category. It includes sampling/gather operations which
    # require decoded A6xx texture/sampler descriptors, plus derivative/quad operations whose
    # semantics depend on lane neighborhoods. No implementation should fake these as an ALU
    # pass-through. Keep this boundary explicit until those execution models exist.
    raise NotImplementedError(f"Category 5 instruction {name!r} requires texture descriptors or quad/subgroup execution semantics")


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
        stores = []
        for i in range(count):
            dst_i = _qcom_reg_component_offset_token(dst, i) if count > 1 else dst
            addr_i = addr + _c(i * dtype.itemsize, addr.dtype)
            value = _load_mem_value(ctx, mem_kind, addr_i, dtype_suffix, lane)
            stores.append(_write_qcom_store(ctx, dst_i, lane, value))
        return UOp.sink(*stores)

    if op.startswith("st"):
        space, addr = _qcom_read_mem_address(ctx, inst.fields[1].value, lane)
        if space != mem_kind: raise RuntimeError(f"{name} uses {mem_kind}[...] but operand addresses {space}[...]")
        src_token = inst.fields[2].value
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


def _compile_inst(inst, ctx):
    # (rptN) executes the instruction N+1 times, advancing dst and every (r)-flagged source by one component each time.
    # That expansion is not modelled yet, and ignoring it would silently compute only the first element.
    if inst.fields and _inst_name(inst).strip().lower() != "nop" and any(re.fullmatch(r"rpt[1-9]\d*", m) for m in inst.fields[0].modifiers):
        raise NotImplementedError(f"(rpt) repeat on {_inst_name(inst)!r} is not implemented")
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
    rngs = [u for u in sink.toposort() if u.op is Ops.RANGE]
    assert len(rngs) <= 1, f"expected at most one lane range, got {len(rngs)}"
    return UOp.sink(sink.end(*rngs)) if rngs else sink

@functools.cache
def _get_runner(inst, merged=False):
    ctx = _Ctx(merged=merged)
    sink = _end_lane_range(_compile_inst(inst, ctx)).replace(arg=KernelInfo(name=f"qcom_{_inst_name(inst)}"))
    with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
        prg = to_program(sink, Device['CPU'].renderer)
        runtime = get_runtime('CPU', prg)
    return prg, runtime

def _compile_at(pc, decoded_shader, merged=False):
    inst = decoded_shader[pc]
    try: return _get_runner(inst, merged), inst
    except Exception as e:
        try: inst_str = repr(inst)
        except Exception: inst_str = f"<{type(inst).__name__}>"
        raise RuntimeError(f"[emu] Failed to compile {inst_str}: {type(e).__name__}: {e}") from e


class WaveState:
    __slots__ = ("wave_size", "wave_idx", "r_reg", "s_reg", "c_reg", "a_reg", "p_reg", "_r_reg_mv", "_s_reg_mv", "_c_reg_mv", "_hr_reg_mv", "_hs_reg_mv", "_hc_reg_mv", "_a_reg_mv", "_p_reg_mv", "pc", "exec_mask", "exec_mask_buf", "_exec_mask_mv", "lane_mask", "predication_mask", "predication_mode", "barrier_waiting", "call_stack", "mergedregs")

    def __init__(
        self, wave_size, wave_idx,
        shader_constants, wgidconstid, wgsizeconstid, localidregid,
        wgid, wgsize, local_dims, mergedregs=False
    ):
        self.wave_size = wave_size
        self.mergedregs = mergedregs
        self.wave_idx = wave_idx
        active_lanes = min(wave_size, max(0, math.prod(local_dims) - wave_idx * wave_size))
        self.lane_mask = (1 << active_lanes) - 1
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
        if wgidconstid != 0xFC:
            for i in range(3):
                self._write_c_reg((wgidconstid + i) >> 2, (wgidconstid + i) & 3, wgid[i])
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
        mask = 0
        for lane in range(self.wave_size):
            if self._p_reg_mv[lane * NUM_P_REG + reg_num * 4 + reg_dim]: mask |= 1 << lane
        return mask

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
_QCOM_OPERAND_VALUE_MODIFIERS = {"neg", "n", "abs", "a", "bnot", "not"}
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
        if _QCOM_FLOAT_IMM_RE.fullmatch(mod) and match.end() == len(expr): return mod, mods
        if mod in _QCOM_OPERAND_VALUE_MODIFIERS or mod in _QCOM_RESULT_FLAGS or re.fullmatch(r"rpt\d+", mod):
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


def _execute_cat0_branch(inst: Inst, state: WaveState, pc: int, jump_locations: dict[str, int], shader_len: int) -> int:
    """Execute the branch subset needed by compute shaders; return the next PC."""
    name = _inst_name(inst).strip().lower()
    if name == "ret":
        if not state.call_stack: raise RuntimeError(f"IR3 ret at PC {pc} with an empty call stack")
        return state.call_stack.pop()
    if name == "call":
        if len(inst.fields) < 2: raise RuntimeError(f"IR3 call missing target: {inst!r}")
        state.call_stack.append(pc + 1)
        return _branch_target(inst.fields[-1].value, pc, jump_locations, shader_len)
    if name == "jump":
        if len(inst.fields) < 2: raise RuntimeError(f"IR3 jump missing target: {inst!r}")
        return _branch_target(inst.fields[-1].value, pc, jump_locations, shader_len)

    # br is divergent in general. This scalar-PC wave scheduler can handle uniform
    # conditions; for bany/ball the reduction itself is uniform. Divergent branches
    # require a reconvergence/branch-stack model and must not be approximated.
    pred_fields = [f.value for f in inst.fields[1:-1] if f.value.lstrip("!").startswith("p")]
    if not pred_fields: raise RuntimeError(f"IR3 {name} missing predicate operand(s): {inst!r}")
    masks = [_predicate_mask(state, tok) for tok in pred_fields]
    active = state.exec_mask & state.lane_mask
    if name == "br":
        true_lanes = masks[0] & active
        false_lanes = active & (~masks[0] & state.lane_mask)
        if true_lanes and false_lanes:
            raise NotImplementedError(f"Divergent br at PC {pc} requires branch reconvergence")
        take = bool(true_lanes)
    elif name == "bany": take = bool(masks[0] & active)
    elif name == "ball": take = bool(active) and (masks[0] & active) == active
    elif name in {"brao", "braa"}:
        if len(masks) != 2: raise RuntimeError(f"IR3 {name} expects two predicates: {inst!r}")
        combined = (masks[0] | masks[1]) if name == "brao" else (masks[0] & masks[1])
        true_lanes, false_lanes = combined & active, active & (~combined & state.lane_mask)
        if true_lanes and false_lanes:
            raise NotImplementedError(f"Divergent {name} at PC {pc} requires branch reconvergence")
        take = bool(true_lanes)
    else: raise NotImplementedError(f"IR3 branch {name!r} is not implemented")
    return _branch_target(inst.fields[-1].value, pc, jump_locations, shader_len) if take else pc + 1


def _barrier_resume_ready(waves: list[WaveState]) -> bool:
    if not waves: return False
    return all(w.barrier_waiting for w in waves)


def run_asm(
    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
    global_dims, group_dims, local_dims, kerneldim, wave_size,
    shared_memsize, pvt_memsize_peritem, pvt_mem_base,
    wgidconstid, wgsizeconstid, wgoffsetconstid, localidregid, linearlocalidregid, mergedregs=False
):
    if wave_size != 64:
        raise NotImplementedError(f"QCOM UOp runner currently supports wave64 only, got wave_size={wave_size}")
    decoded_shader, jump_locations = _decode_program(shader)

    program: dict[int, tuple[Callable, list[int], bool, Inst]] = {}

    total_threads = math.prod(local_dims)

    vmem_buf = Buffer("CPU", 1<<48, options=BufferSpec(external_ptr=0)).ensure_allocated()
    # Private-memory address rebasing is not implemented yet. Preserve the launch-provided
    # base below, but do not allocate a private_buf that is never used by the runner.
    has_private_memory = pvt_memsize_peritem > 0

    def _ensure_compiled(pc):
        if pc not in program:
            (prg, runtime), inst = _compile_at(pc, decoded_shader, mergedregs)
            is_barrier = _is_barrier_inst(inst)
            program[pc] = (runtime.fxn, prg.arg.globals, is_barrier, inst)
            if DEBUG >= 3:
                prev_len = _get_runner.cache_info().currsize
                msg = f"[emu] PC={pc}: {inst!r}"
                print(colored(msg, 'green') if _get_runner.cache_info().currsize > prev_len else msg)
        return program[pc]

    def _run_workgroup(group_id):
        shared_buf = Buffer('CPU', max(shared_memsize, 1)).ensure_allocated()
        ctypes.memset(shared_buf._buf, 0, max(shared_memsize, 1))
        waves = []
        for wave_start in range(0, total_threads, wave_size):
            st = _init_wave(
                shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
                wave_start, wave_size, total_threads, local_dims, pvt_memsize_peritem,
                wgidconstid, wgsizeconstid, localidregid, group_id, group_dims, mergedregs
            )
            waves.append(
                (st,
                    [
                        ctypes.c_uint64(st.r_reg._buf),
                        ctypes.c_uint64(st.s_reg._buf),
                        ctypes.c_uint64(st.c_reg._buf),
                        ctypes.c_uint64(st.a_reg._buf),
                        ctypes.c_uint64(st.p_reg._buf),
                        ctypes.c_uint64(vmem_buf._buf),
                        ctypes.c_uint64(shared_buf._buf),
                        ctypes.c_uint64(pvt_mem_base if has_private_memory else 0),
                        ctypes.c_uint64(st.exec_mask_buf._buf)
                    ]
                )
            )
        done = [False] * len(waves)
        for _ in range(10_000_000):
            if all(done): return
            active_waves = [w for wi, w in enumerate(waves) if not done[wi]]
            if any(st.barrier_waiting for st, _ in active_waves):
                # A workgroup barrier is only valid when every wave still participating
                # in the workgroup reaches it. If another wave has already ended, do not
                # accidentally release the waiting waves as though that wave had arrived.
                if any(done):
                    raise RuntimeError("workgroup reached a barrier after another wave had already completed")
                if _barrier_resume_ready([st for st, _ in active_waves]):
                    for st, _ in active_waves:
                        st.barrier_waiting = False
            for wi, (st, bufs) in enumerate(waves):
                if done[wi] or st.barrier_waiting: continue
                for _ in range(1_000_000):
                    pc = st.pc
                    if not 0 <= pc < len(decoded_shader):
                        raise RuntimeError(f"wave {wi} reached PC {pc} outside shader ({len(decoded_shader)} instructions)")
                    st.pc += 1 # each instruction is 2 words
                    inst = decoded_shader[pc]
                    if DEBUG >= 5: print(f"  exec gid=({group_id[0]},{group_id[1]},{group_id[2]}) w={wi} PC={pc}: {inst!r}", flush=True)
                    if _is_cat0_branch(inst):
                        st.pc = _execute_cat0_branch(inst, st, pc, jump_locations, len(decoded_shader))
                        continue
                    fxn, globals_list, is_barrier, inst = _ensure_compiled(pc)
                    st._exec_mask_mv[0] = st.exec_mask & 0xFFFFFFFFFFFFFFFF
                    fxn((ctypes.c_uint64 * len(globals_list))(*[bufs[g].value for g in globals_list]))
                    if _is_predication_enable_inst(inst) or _is_predication_disable_inst(inst):
                        _apply_wave_predication(inst, st)
                    if _is_end_inst(inst):
                        done[wi] = True
                        break
                    if is_barrier:
                        st.barrier_waiting = True
                        break
                else: raise RuntimeError("exceeded 1M instructions in single wave, likely infinite loop")
        raise RuntimeError("exceeded 10M total scheduling rounds")
    (group_x, group_y, group_z) = group_dims

    for gidz, gidy, gidx in itertools.product(range(group_z), range(group_y), range(group_x)):
        _run_workgroup((gidx, gidy, gidz))