import ctypes
from dataclasses import dataclass
import functools
import itertools
import math
from typing import Callable

from tinygrad.codegen import to_program
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.engine.realize import get_runtime
from tinygrad.helpers import DEBUG, Context, ceildiv, colored
from tinygrad.uop.ops import UOp, Ops
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

def _c(val, dtype=dtypes.uint32): return UOp.const(val, dtype)

@dataclass(frozen=True)
class Inst:
    ...
        
        

class _Ctx:
    def __init__(self):
        return

def _compile_inst(inst, ctx):
    ...

@functools.cache
def _get_runner(inst):
    ctx = _Ctx() #TODO: tbd
    sink = _compile_inst(inst, ctx)
    with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
        prg = to_program(sink, Device['CPU'].renderer)
        runtime = get_runtime('CPU', prg)
    return prg, runtime
def _decode_inst(inst_code):
    #TODO: create instruction class and decode based on bits
    category = (inst_code >> 61) & 0x7
    if category == 0:
        immediate = inst_code & 0xFFFFFFFF
        repeat = (inst_code >> 40) & 0x7
        ss = (inst_code >> 44) & 0x1
        eq = (inst_code >> 48) & 0x1
        jp = (inst_code >> 59) & 0x1
        sy = (inst_code >> 60) & 0x1

    elif category == 1: pass
    elif category == 2: pass
    elif category == 3: pass
    elif category == 4: pass
    elif category == 5: pass
    elif category == 6: pass
    elif category == 7: pass
    else: raise RuntimeError(f"Invalid instruction category: {category}")
    return Inst(inst_code)

def _decode_at(pc, shader):
    inst_low = shader[pc]
    inst_high = shader[pc + 1]
    inst_code = (inst_high << 32) | inst_low
    inst = _decode_inst(inst_code)
    try: return _get_runner(inst), inst
    except Exception as e:
        try: inst_str = repr(inst)
        except Exception: inst_str = f"<{type(inst).__name__}>"
        raise RuntimeError(f"[emu] Failed to compile {inst_str}: {type(e).__name__}: {e}") from e
    

class WaveState:
    #TODO: regs may be better placed elsewhere (_Ctx, and as buffers passed in in run_asm)
    __slots__ = {"wave_size", "wave_idx", "regs", "_axis_id", "pc"}
    #cache of some sort?

    def _decode_reg(self, reg_id):
        return reg_id >> 2, reg_id & 3 

    def get_reg(self, reg_type, reg_num, reg_dim):
        return self.regs[(reg_type, reg_num, reg_dim)]

    def set_reg(self, reg_type, reg_num, reg_dim, value):
        self.regs[(reg_type, reg_num, reg_dim)] = value

    def range(self, n = None):
        if n is None: n = self.wave_size
        self._axis_id += 1
        return UOp.range(n, self._axis_id, dtype=dtypes.int)

    def __init__(
        self, wave_size, wave_idx,
        wgidconstid, wgsizeconstid, localidregid,
        wgid, wgsize, local_dims
    ):
        self.wave_size = wave_size
        self.wave_idx = wave_idx
        self._axis_id = 0
        self.regs = {}
        self.pc = 0

        wgid_reg, _ = self._decode_reg(wgidconstid)
        for i in range(3):
            self.regs[("c", wgid_reg, i)] = _c(wgid[i])
        wgsize_reg, _ = self._decode_reg(wgsizeconstid)
        for i in range(3):
            self.regs[("c", wgsize_reg, i)] = _c(wgsize[i])

        localid_reg, _ = self._decode_reg(localidregid)
        lane = self.range(wave_size) + _c(wave_idx * wave_size)
        local_x = lane % _c(local_dims[0])
        local_y = (lane // _c(local_dims[0])) % _c(local_dims[1])
        local_z = lane // (_c(local_dims[0]) * _c(local_dims[1]))
        localid = (local_x, local_y, local_z)
        for i in range(3):
            self.regs[("r", localid_reg, i)] = localid[i]
        #wgoffset and linearlocalid not exposed

def _init_wave(
    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
    wave_start, wave_size, total_threads, local_dims, pvt_memsize_peritem,
    wgidconstid, wgsizeconstid, localidregid, group_id, group_dims
):
    return WaveState(
        wave_size=wave_size,
        wave_idx=wave_start // wave_size,
        wgidconstid=wgidconstid,
        wgsizeconstid=wgsizeconstid,
        localidregid=localidregid,
        wgid=group_id,
        wgsize=group_dims,
        local_dims=local_dims
    )

def run_asm(
    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
    global_dims, group_dims, local_dims, kerneldim, wave_size,
    shared_memsize, pvt_memsize_peritem, pvt_mem_base,
    wgidconstid, wgsizeconstid, wgoffsetconstid, localidregid, linearlocalidregid
):

    program: dict[int, tuple[Callable, list[int], bool, Inst]] = {} #TODO: clarify types

    total_threads = math.prod(local_dims)


    vmem_buf = Buffer('CPU', 1<<40, dtypes.uint8, options=BufferSpec(external_ptr=0)).ensure_allocated()
    shared_buf = Buffer('CPU', max(shared_memsize, 1), dtypes.uint8).ensure_allocated()

    n_waves = ceildiv(total_threads, wave_size)
    #TODO remember to subtract pvt_mem_base from private mem accesses
    private_buf = Buffer('CPU', pvt_memsize_peritem * wave_size * n_waves, dtypes.uint8).ensure_allocated() if pvt_memsize_peritem > 0 else None

    def _ensure_compiled(pc):
        if pc not in program:
            (prg, runtime), inst = _decode_at(pc, shader)
            is_barrier = ... #TODO: determine barrier instructions
            program[pc] = (runtime.fxn, prg.arg.globals, is_barrier, inst)
            if DEBUG >= 3:
                prev_len = _get_runner.cache_info().currsize
                msg = f"[emu] PC={pc}: {inst!r}"
                print(colored(msg, 'green') if _get_runner.cache_info().currsize > prev_len else msg)
        return program[pc]

    def _run_workgroup(group_id):
        waves = []
        for wave_start in range(0, total_threads, wave_size):
            st = _init_wave(
                shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
                wave_start, wave_size, total_threads, local_dims, pvt_memsize_peritem,
                wgidconstid, wgsizeconstid, localidregid, group_id, group_dims
            )
            waves.append(
                (st,
                    [
                        ctypes.c_uint64(vmem_buf._buf),
                        ctypes.c_uint64(shared_buf._buf),
                        ctypes.c_uint64(pvt_mem_base if private_buf else 0)
                    ]
                )
            )
        done = [False] * len(waves)
        for _ in range(10_000_000):
            if all(done): return
            for wi, (st, bufs) in enumerate(waves):
                if done[wi]: continue
                for _ in range(1_000_000):
                    pc = st.pc
                    st.pc += 2 #each instruction is 2 words
                    fxn, globals_list, is_barrier, inst = _ensure_compiled(pc)    
                    #TODO done detection? probably using end opcode, or instruction list size
                    if DEBUG >= 5: print(f"  exec gid=({group_id[0]},{group_id[1]},{group_id[2]}) w={wi} PC={pc}: {inst!r}", flush=True)
                    fxn(*[bufs[g] for g in globals_list])
                    if is_barrier: break
                else: raise RuntimeError("exceeded 1M instructions in single wave, likely infinite loop")
        raise RuntimeError("exceeded 10M total scheduling rounds")
    (group_x, group_y, group_z) = group_dims

    for gidz, gidy, gidx in itertools.product(range(group_z), range(group_y), range(group_x)):
        _run_workgroup((gidx, gidy, gidz))