import ctypes
import functools

from test.mockgpu.driver import TextFileDesc, VirtDriver, VirtFile, VirtFileDesc

from test.mockgpu.qcom.qcomgpu import QCOMGPU, QCOMCommandObject
from tinygrad.runtime.autogen import kgsl

import functools

_UINT_MAX = (1 << (8*ctypes.sizeof(ctypes.c_uint))) - 1

def _ioctl_nr(ioctl: functools.partial) -> int: return ioctl.args[2]

kgsl_ioctl_info = {
    _ioctl_nr(ioctl): (name ,ioctl.args[3]) for name, ioctl in vars(kgsl).items()
    if name.startswith("IOCTL_KGSL_") and isinstance(ioctl, functools.partial)
}

class KGSLFileDesc(VirtFileDesc):
    def __init__(self, fd, driver):
        super().__init__(fd)
        self.driver = driver
    def ioctl(self, fd, request, argp): return self.driver.kgsl_ioctl(request, argp)
    def mmap(self, start, sz, prot, flags, fd, offset):
        obj_id = offset // 0x1000
        if obj_id not in self.driver.object_by_id: raise RuntimeError(f"invalid object id: {obj_id}")
        return self.driver.object_by_id[obj_id].vaddr




class QCOMContext:
    #TODO flag implementation and (increasingly unlikely) reference counting for invalidation
    def __init__(self, flags, gpu):
        self.flags = self._validate_flags(flags)
        self.gpu = gpu
        self.timestamp = 1 #always per context
        self.submissions = {}
    def _validate_flags(self, flags):
        #device preemption, secure context supported on a630, so we skip those checks
        flags = flags & (
            kgsl.KGSL_CONTEXT_PREAMBLE |
            kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC |
            kgsl.KGSL_CONTEXT_PER_CONTEXT_TS |
            kgsl.KGSL_CONTEXT_USER_GENERATED_TS |
            kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE |
            kgsl.KGSL_CONTEXT_CTX_SWITCH |
            kgsl.KGSL_CONTEXT_PRIORITY_MASK |
            kgsl.KGSL_CONTEXT_TYPE_MASK |
            kgsl.KGSL_CONTEXT_PWR_CONSTRAINT |
            kgsl.KGSL_CONTEXT_IFH_NOP |
            kgsl.KGSL_CONTEXT_SECURE |
            kgsl.KGSL_CONTEXT_PREEMPT_STYLE_MASK
        ) #KGSL_CONTEXT_INVALIDATE_ON_FAULT, KGSL_CONTEXT_NO_SNAPSHOT, KGSL_CONTEXT_SPARSE not found in kgsl.py
        assert (flags & kgsl.KGSL_CONTEXT_PREAMBLE) and (flags & kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC), "legacy context switch not supported on a630"
        return flags
    def submit_command(self, numcmds, cmdlist, cmdsize):
        cmd_ids = [0] * numcmds
        self.gpu.submit_command(numcmds, cmdlist, cmdsize)
        allocated_timestamp = self.timestamp
        self.submissions[allocated_timestamp] = cmd_ids
        self.timestamp += 1
        return allocated_timestamp
        

class QCOMObject:
    def __init__(self, vaddr, data, size):
        self.vaddr = vaddr
        self.data = data
        self.size = size

class QCOMDriver(VirtDriver):
    def __init__(self):
        super().__init__()
        self.tracked_files += [VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self))] + \
            [VirtFile('/sys/class/kgsl/kgsl-3d0/idle_timer', functools.partial(TextFileDesc, text=str(_UINT_MAX+1)))]

        self.gpu = QCOMGPU(1)

        self.next_fd = (1 << 30)

        self.next_ctx_id = 1
        self.context_by_id = {}
        
        self.next_obj_id = 1
        self.object_by_id = {}

    def _alloc_fd(self):
        my_fd = self.next_fd
        self.next_fd = self.next_fd + 1
        return my_fd
    def open(self, name, flags, mode, fdcls): return fdcls.fdcls(self._alloc_fd())
    def _decode_request(self, request):
        out = [0]*4
        for i in range(4):
            out[4-i-1] = (request >> (i * 8)) & 0xFF
        return out
    def kgsl_ioctl(self, request, argp):
        (dir, size, typ, nr) = self._decode_request(request)
        assert typ == 0x09, "Expected type 0x09 for KGSL ioctl"

        if nr not in kgsl_ioctl_info: raise RuntimeError(f"unknown kgsl ioctl, {nr} unknown")
        name, struct_type = kgsl_ioctl_info[nr]
        struct = struct_type.from_address(argp)

        if nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
            if struct.type == kgsl.KGSL_PROP_DEVICE_INFO:
                info = ctypes.cast(
                    struct.value,
                    ctypes.POINTER(kgsl.struct_kgsl_devinfo)
                )
                info.contents.device_id = self.gpu.devinfo["device_id"]
                info.contents.chip_id = self.gpu.devinfo["chip_id"]
                info.contents.mmu_enabled = int(self.gpu.devinfo["mmu_enabled"])
                info.contents.gmem_gpubaseaddr = self.gpu.devinfo["gmem_gpubaseaddr"]
                info.contents.gpu_id = self.gpu.devinfo["gpu_id"]
                info.contents.gmem_sizebytes = self.gpu.devinfo["gmem_sizebytes"]
                struct.sizebytes = kgsl.struct_kgsl_devinfo.SIZE
            else: raise NotImplementedError(f"property id {struct.propid} not implemented")
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID):
            ctx_id = struct.context_id
            if ctx_id not in self.context_by_id: raise RuntimeError(f"invalid context id: {ctx_id}")
            ctx = self.context_by_id[ctx_id]
            timestamp = struct.timestamp
            timeout = struct.timeout
            if timestamp >= ctx.timestamp:
                raise RuntimeError(f"invalid timestamp: {timestamp}") #synchronous execution
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE):
            flags = struct.flags
            ctx = QCOMContext(flags, self.gpu)
            ctx_id = self.next_ctx_id
            self.context_by_id[ctx_id] = ctx
            struct.drawctxt_id = ctx_id
            self.next_ctx_id += 1
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY):
            ctx_id = struct.drawctxt_id
            if ctx_id not in self.context_by_id: raise RuntimeError(f"invalid context id: {ctx_id}")
            del self.context_by_id[ctx_id]
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_MAP_USER_MEM):
            if struct.memtype != kgsl.KGSL_USER_MEM_TYPE_ADDR: raise NotImplementedError("Only KGSL_USER_MEM_TYPE_ADDR is supported")
            self.gpu.map_range(struct.hostptr+struct.offset, struct.len)
            struct.gpuaddr = struct.hostptr+struct.offset
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID):
            ctx_id = struct.context_id
            if ctx_id not in self.context_by_id: raise RuntimeError(f"invalid context id: {ctx_id}")
            typ = struct.type
            if typ != kgsl.KGSL_TIMESTAMP_QUEUED: raise NotImplementedError("Only KGSL_TIMESTAMP_QUEUED is supported")
            ctx = self.context_by_id[ctx_id]
            struct.timestamp = ctx.timestamp - 1 #queued
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP_CTXTID):
            ctx_id = struct.context_id
            if ctx_id not in self.context_by_id: raise RuntimeError(f"invalid context id: {ctx_id}")
            ctx = self.context_by_id[ctx_id]
            struct.timestamp = ctx.timestamp - 1 #queued
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SHAREDMEM_FREE):
            self.gpu.unmap_range(struct.gpuaddr, struct.len)
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SETPROPERTY): pass #only used to set power constraint - irrelevant for emulator
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
            size = struct.size
            if size == 0 or size > _UINT_MAX: raise RuntimeError(f"size: {size} out of range (0, UINT_MAX]")
            flags = struct.flags
            # ignore va_len, metadata, unused
            if not (flags & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP): raise NotImplementedError("Only KGSL_MEMFLAGS_USE_CPU_MAP is supported") 
            data = bytearray(size) #map to cpu address, then map to gpu too
            vaddr = ctypes.addressof(ctypes.c_char.from_buffer(data))
            self.gpu.map_range(vaddr, size)
            self.object_by_id[self.next_obj_id] = QCOMObject(vaddr, data, size)
            struct.id = self.next_obj_id
            self.next_obj_id += 1
            struct.mmapsize = size
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_FREE):
            obj_id = struct.id
            if obj_id not in self.object_by_id: raise RuntimeError(f"invalid object id: {obj_id}")
            obj = self.object_by_id[obj_id]
            self.gpu.unmap_range(obj.vaddr, obj.size)
            del self.object_by_id[obj_id]
        elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND):
            #flags not used
            ctx_id = struct.context_id
            ctx = self.context_by_id[ctx_id] #i.e., throw error if fails
            struct.timestamp = self.context_by_id[ctx_id].submit_command(struct.numcmds, struct.cmdlist, struct.cmdsize)
        else: raise NotImplementedError(f"{name} not implemented")
        return 0
