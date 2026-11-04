# ruff: noqa: E501
import time

from test.mockgpu.gpu import VirtGPU
from test.mockgpu.qcom.emu import run_asm
from tinygrad.runtime.autogen import kgsl, mesa
import ctypes


class QCOMCommandObject:
    def __init__(self, numcmds, cmdlist, cmdsize):
        self.numcmds = numcmds
        self.cmdlist = cmdlist
        self.cmdsize = cmdsize

class QCOMGPURegisters:
    #admittedly overkill for what is just a dictionary wrapper, but just in case functionality needs to grow
    def __init__(self):
        self.regs = {}
    def __getitem__(self, key):
        return self.regs[key]
    def __setitem__(self, key, value):
        self.regs[key] = value

class QCOMGPU(VirtGPU):
    def __init__(self, gpu_id):
        super().__init__(gpu_id)
        self.arch = "a630"
        self.devinfo = {
            "device_id": 0xC0DEFACE,
            "chip_id": 0x06030000,  # a630 emulator (6, 3, 0, 0)
            "mmu_enabled": 1,
            "gmem_gpubaseaddr": 0x00100000,
            "gpu_id": gpu_id,
            "gmem_sizebytes": 0x00100000
        }
        self.state = {}
        self.state_to_unit = {
            (mesa.SB6_CS_SHADER, mesa.ST6_SHADER): 128,
            (mesa.SB6_CS_SHADER, mesa.ST6_CONSTANTS): 16,
            (mesa.SB6_CS_SHADER, mesa.ST6_UAV): 64,
            (mesa.SB6_CS_TEX, mesa.ST6_SHADER): 16,
            (mesa.SB6_CS_TEX, mesa.ST6_CONSTANTS): 64, #one 16-dword A6XX_TEX_CONST descriptor per unit
        }
        self.cs_tex_const_size = 1600
        self.mapped_ranges = set()
        self.queue = []
        self.regs = QCOMGPURegisters()

    def map_range(self, vaddr, size):
        self.mapped_ranges.add((vaddr, size))
    def unmap_range(self, vaddr, size):
        self.mapped_ranges.remove((vaddr, size))

    def _decode_pkt7(self, pkt7):
        opcode = (pkt7 >> 16) & 0x7F
        cnt = pkt7 & 0x3FFF
        return opcode, cnt

    def _decode_pkt4(self, pkt4):
        reg = (pkt4 >> 8) & 0x3FFFF
        cnt = pkt4 & 0x7F
        return reg, cnt

    def _read_word(self, addr, signed=False):
        if signed:
            return ctypes.c_int32.from_address(addr).value
        else:
            return ctypes.c_uint32.from_address(addr).value

    def _write_word(self, addr, value):
        ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint32))
        ptr.contents.value = value

    def _handle_pkt7(self, opcode, payload):
        if opcode == mesa.CP_EVENT_WRITE:
            event_code = (payload[0] & mesa.CP_EVENT_WRITE_0_EVENT__MASK) >> mesa.CP_EVENT_WRITE_0_EVENT__SHIFT
            timestamp = (payload[0] & mesa.CP_EVENT_WRITE_0_TIMESTAMP) != 0
            irq = (payload[0] & mesa.CP_EVENT_WRITE_0_IRQ) != 0
            if timestamp or irq: raise NotImplementedError("CP_EVENT_WRITE with timestamp/irq flags set not implemented")
            if event_code == mesa.CACHE_FLUSH_TS:
                addr_low = (payload[1] & mesa.CP_EVENT_WRITE_1_ADDR_0_LO__MASK) >> mesa.CP_EVENT_WRITE_1_ADDR_0_LO__SHIFT
                addr_high = (payload[2] & mesa.CP_EVENT_WRITE_2_ADDR_0_HI__MASK) >> mesa.CP_EVENT_WRITE_2_ADDR_0_HI__SHIFT
                addr = (addr_high << 32) | addr_low
                value = payload[3]
                self._write_word(addr, value)
            elif event_code == mesa.CACHE_INVALIDATE: pass #no observable effects
            else: raise NotImplementedError(f"CP_EVENT_WRITE with event_code {event_code:#X} not implemented")
        elif opcode == mesa.CP_WAIT_MEM_WRITES: pass
        elif opcode == mesa.CP_WAIT_FOR_IDLE: pass
        elif opcode == mesa.CP_REG_TO_MEM:
            reg_code = (payload[0] & mesa.CP_REG_TO_MEM_0_REG__MASK) >> mesa.CP_REG_TO_MEM_0_REG__SHIFT
            if reg_code == mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER:
                clock = int(time.perf_counter()*19.2*1e6) #19.2 MHz for always_on_counter
                self.regs[mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER] = clock & 0xFFFFFFFF
                self.regs[mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER+1] = (clock >> 32) & 0xFFFFFFFF
            cnt = (payload[0] & mesa.CP_REG_TO_MEM_0_CNT__MASK) >> mesa.CP_REG_TO_MEM_0_CNT__SHIFT
            # _64b = (payload[0] & mesa.CP_REG_TO_MEM_0_64B) != 0
            dest_addr_low = (payload[1] & mesa.CP_REG_TO_MEM_1_DEST__MASK) >> mesa.CP_REG_TO_MEM_1_DEST__SHIFT
            dest_addr_high = (payload[2] & mesa.CP_REG_TO_MEM_2_DEST_HI__MASK) >> mesa.CP_REG_TO_MEM_2_DEST_HI__SHIFT
            dest_addr = (dest_addr_high << 32) | dest_addr_low
            for i in range(cnt):
                reg_val = self.regs[reg_code + i]
                self._write_word(dest_addr + i * ctypes.sizeof(ctypes.c_uint32), reg_val) #little endian, so we can ignore _64b flag, since treatment is identical
        elif opcode == mesa.CP_WAIT_REG_MEM:
            function_code = (payload[0] & mesa.CP_WAIT_REG_MEM_0_FUNCTION__MASK) >> mesa.CP_WAIT_REG_MEM_0_FUNCTION__SHIFT
            signed_compare = (payload[0] & mesa.CP_WAIT_REG_MEM_0_SIGNED_COMPARE) != 0 #ignore, unused
            poll_code = (payload[0] & mesa.CP_WAIT_REG_MEM_0_POLL__MASK) >> mesa.CP_WAIT_REG_MEM_0_POLL__SHIFT
            poll_addr_low = (payload[1] & mesa.CP_WAIT_REG_MEM_1_POLL_ADDR_LO__MASK) >> mesa.CP_WAIT_REG_MEM_1_POLL_ADDR_LO__SHIFT
            poll_addr_high = (payload[2] & mesa.CP_WAIT_REG_MEM_2_POLL_ADDR_HI__MASK) >> mesa.CP_WAIT_REG_MEM_2_POLL_ADDR_HI__SHIFT
            poll_addr = (poll_addr_high << 32) | poll_addr_low
            ref_code = (payload[3] & mesa.CP_WAIT_REG_MEM_3_REF__MASK) >> mesa.CP_WAIT_REG_MEM_3_REF__SHIFT
            mask_code = (payload[4] & mesa.CP_WAIT_REG_MEM_4_MASK__MASK) >> mesa.CP_WAIT_REG_MEM_4_MASK__SHIFT
            ref_val = ref_code & mask_code
            if signed_compare:
                ref_val = ctypes.c_int32(ref_val).value & mask_code
            if poll_code == mesa.POLL_MEMORY:
                poll_val = self._read_word(poll_addr, signed=signed_compare) & mask_code
            elif poll_code == mesa.POLL_REGISTER:
                if signed_compare:
                    poll_val = ctypes.c_int32(self.regs[poll_addr_low]).value & mask_code
                else:
                    poll_val = self.regs[poll_addr_low] & mask_code
            else: raise NotImplementedError(f"CP_WAIT_REG_MEM with poll_code {poll_code:#X} not implemented")
            success = False
            if function_code == mesa.WRITE_EQ:
                success = poll_val == ref_val
            elif function_code == mesa.WRITE_GE:
                success = poll_val >= ref_val
            elif function_code == mesa.WRITE_GT:
                success = poll_val > ref_val
            elif function_code == mesa.WRITE_LE:
                success = poll_val <= ref_val
            elif function_code == mesa.WRITE_LT:
                success = poll_val < ref_val
            elif function_code == mesa.WRITE_NE:
                success = poll_val != ref_val
            else: raise RuntimeError(f"unknown function code {function_code:#X}")
            if not success: raise RuntimeError("CP_WAIT_REG_MEM failed")
        elif opcode == mesa.CP_SET_MARKER:
            mode_code = (payload[0] & mesa.A6XX_CP_SET_MARKER_0_MODE__MASK) >> mesa.A6XX_CP_SET_MARKER_0_MODE__SHIFT
            if mode_code != mesa.RM6_COMPUTE: raise NotImplementedError(f"CP_SET_MARKER with mode_code {mode_code:#X} not implemented")
        elif opcode == mesa.CP_LOAD_STATE6_FRAG:
            dst_off = (payload[0] & mesa.CP_LOAD_STATE6_0_DST_OFF__MASK) >> mesa.CP_LOAD_STATE6_0_DST_OFF__SHIFT
            if dst_off != 0: raise NotImplementedError("CP_LOAD_STATE6_FRAG with non-zero dst_off not implemented")
            state_type_code = (payload[0] & mesa.CP_LOAD_STATE6_0_STATE_TYPE__MASK) >> mesa.CP_LOAD_STATE6_0_STATE_TYPE__SHIFT
            state_src_code = (payload[0] & mesa.CP_LOAD_STATE6_0_STATE_SRC__MASK) >> mesa.CP_LOAD_STATE6_0_STATE_SRC__SHIFT
            if state_src_code != mesa.SS6_INDIRECT: raise NotImplementedError(f"CP_LOAD_STATE6_FRAG with state_src_code {state_src_code:#X} not implemented")
            state_block_code = (payload[0] & mesa.CP_LOAD_STATE6_0_STATE_BLOCK__MASK) >> mesa.CP_LOAD_STATE6_0_STATE_BLOCK__SHIFT
            num_unit = (payload[0] & mesa.CP_LOAD_STATE6_0_NUM_UNIT__MASK) >> mesa.CP_LOAD_STATE6_0_NUM_UNIT__SHIFT
            ext_src_addr_low = payload[1] & mesa.CP_LOAD_STATE6_1_EXT_SRC_ADDR__MASK
            ext_src_addr_high = (payload[2] & mesa.CP_LOAD_STATE6_2_EXT_SRC_ADDR_HI__MASK) >> mesa.CP_LOAD_STATE6_2_EXT_SRC_ADDR_HI__SHIFT
            ext_src_addr = (ext_src_addr_high << 32) | ext_src_addr_low
            read_ptr = ext_src_addr
            data = []
            num_words = num_unit * self.state_to_unit.get((state_block_code, state_type_code), 0) // 4 #bytes to words
            if num_words == 0: raise NotImplementedError("CP_LOAD_STATE6_FRAG with unimplemented state_block_code/state_type_code combination")
            if num_words < 0: num_words = self.cs_tex_const_size // 4 #fixed size (according to opencl_ioctl)
            for i in range(num_words):
                data.append(self._read_word(read_ptr))
                read_ptr += ctypes.sizeof(ctypes.c_uint32)
            self.state[(state_block_code, state_type_code)] = data
        elif opcode == mesa.CP_EXEC_CS:
            ngroups_x = (payload[1] & mesa.CP_EXEC_CS_1_NGROUPS_X__MASK) >> mesa.CP_EXEC_CS_1_NGROUPS_X__SHIFT
            ngroups_y = (payload[2] & mesa.CP_EXEC_CS_2_NGROUPS_Y__MASK) >> mesa.CP_EXEC_CS_2_NGROUPS_Y__SHIFT
            ngroups_z = (payload[3] & mesa.CP_EXEC_CS_3_NGROUPS_Z__MASK) >> mesa.CP_EXEC_CS_3_NGROUPS_Z__SHIFT
            self._execute_cs((ngroups_x, ngroups_y, ngroups_z)) #all other info should be in registers and state
        elif opcode == mesa.CP_RUN_OPENCL: raise NotImplementedError("Only CP_EXEC_CS supported for now (IR3Renderer)")
        else: raise NotImplementedError(f"PKT7 opcode {opcode:#X} not implemented")

    def _handle_pkt4(self, reg, cnt, payload):
        for i in range(cnt):
            self.regs[reg + i] = payload[i]

    def _decode_inst(self, inst_ptr):
        inst = ctypes.c_uint32.from_address(inst_ptr).value
        if inst & mesa.CP_TYPE7_PKT == mesa.CP_TYPE7_PKT:
            opcode, cnt = self._decode_pkt7(inst)
            payload = [0] * cnt
            for i in range(cnt):
                inst_ptr += ctypes.sizeof(ctypes.c_uint32)
                payload[i] = self._read_word(inst_ptr)
            self._handle_pkt7(opcode, payload)
        elif inst & mesa.CP_TYPE4_PKT == mesa.CP_TYPE4_PKT:
            reg, cnt = self._decode_pkt4(inst)
            payload = [0] * cnt
            for i in range(cnt):
                inst_ptr += ctypes.sizeof(ctypes.c_uint32)
                payload[i] = self._read_word(inst_ptr)
            self._handle_pkt4(reg, cnt, payload)
        inst_ptr += ctypes.sizeof(ctypes.c_uint32)
        return inst_ptr

    def _execute_cs(self, group_dims):
        ndrange_reg = self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0]
        kerneldim = (ndrange_reg & mesa.A6XX_SP_CS_NDRANGE_0_KERNELDIM__MASK) >> mesa.A6XX_SP_CS_NDRANGE_0_KERNELDIM__SHIFT
        local_x = ((ndrange_reg & mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__MASK) >> mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__SHIFT) + 1
        local_y = ((ndrange_reg & mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__MASK) >> mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__SHIFT) + 1
        local_z = ((ndrange_reg & mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__MASK) >> mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__SHIFT) + 1
        local_dims = (local_x, local_y, local_z)
        #globaloffsets set to 0
        globalsize_x = (self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_1] & mesa.A6XX_SP_CS_NDRANGE_1_GLOBALSIZE_X__MASK) >> mesa.A6XX_SP_CS_NDRANGE_1_GLOBALSIZE_X__SHIFT
        globalsize_y = (self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_3] & mesa.A6XX_SP_CS_NDRANGE_3_GLOBALSIZE_Y__MASK) >> mesa.A6XX_SP_CS_NDRANGE_3_GLOBALSIZE_Y__SHIFT
        globalsize_z = (self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_5] & mesa.A6XX_SP_CS_NDRANGE_5_GLOBALSIZE_Z__MASK) >> mesa.A6XX_SP_CS_NDRANGE_5_GLOBALSIZE_Z__SHIFT
        global_dims = (globalsize_x, globalsize_y, globalsize_z)
        #global_offsets = (0, 0, 0)
        cntl_reg_0 = self.regs[mesa.REG_A6XX_SP_CS_CNTL_0]
        wave_size_enum = (cntl_reg_0 & mesa.A6XX_SP_CS_CNTL_0_THREADSIZE__MASK) >> mesa.A6XX_SP_CS_CNTL_0_THREADSIZE__SHIFT
        wave_size = 64
        if wave_size_enum == mesa.THREAD128: wave_size = 128
        mergedregs = bool(cntl_reg_0 & mesa.A6XX_SP_CS_CNTL_0_MERGEDREGS)  # half regs alias the halves of full regs only when set
        # halfregfootprint = (cntl_reg_0 & mesa.A6XX_SP_CS_CNTL_0_HALFREGFOOTPRINT__MASK) >> mesa.A6XX_SP_CS_CNTL_0_HALFREGFOOTPRINT__SHIFT
        # fullregfootprint = (cntl_reg_0 & mesa.A6XX_SP_CS_CNTL_0_FULLREGFOOTPRINT__MASK) >> mesa.A6XX_SP_CS_CNTL_0_FULLREGFOOTPRINT__SHIFT
        # branchstack = (cntl_reg_0 & mesa.A6XX_SP_CS_CNTL_0_BRANCHSTACK__MASK) >> mesa.A6XX_SP_CS_CNTL_0_BRANCHSTACK__SHIFT
        cntl_reg_1 = self.regs[mesa.REG_A6XX_SP_CS_CNTL_1]
        shared_memsize = (cntl_reg_1 & mesa.A6XX_SP_CS_CNTL_1_SHARED_SIZE__MASK) >> mesa.A6XX_SP_CS_CNTL_1_SHARED_SIZE__SHIFT
        pvt_mem_param_reg = self.regs[mesa.REG_A6XX_SP_CS_PVT_MEM_PARAM]
        pvt_memsize_peritem = (pvt_mem_param_reg & mesa.A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM__MASK) >> mesa.A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM__SHIFT
        pvt_mem_base = self.regs[mesa.REG_A6XX_SP_CS_PVT_MEM_BASE]
        #above fields and those of cntl_reg_1 are somewhat irrelevant for our emulator
        shader = self.state[(mesa.SB6_CS_SHADER, mesa.ST6_SHADER)]
        shader_constants = self.state[(mesa.SB6_CS_SHADER, mesa.ST6_CONSTANTS)]
        shader_uav = self.state.get((mesa.SB6_CS_SHADER, mesa.ST6_UAV), [])
        shader_tex = self.state.get((mesa.SB6_CS_TEX, mesa.ST6_SHADER), [])
        shader_tex_constants = self.state.get((mesa.SB6_CS_TEX, mesa.ST6_CONSTANTS), [])
        const_config_reg = self.regs[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
        wgidconstid = (const_config_reg & mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__MASK) >> mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__SHIFT
        wgsizeconstid = (const_config_reg & mesa.A6XX_SP_CS_CONST_CONFIG_0_WGSIZECONSTID__MASK) >> mesa.A6XX_SP_CS_CONST_CONFIG_0_WGSIZECONSTID__SHIFT
        wgoffsetconstid = (const_config_reg & mesa.A6XX_SP_CS_CONST_CONFIG_0_WGOFFSETCONSTID__MASK) >> mesa.A6XX_SP_CS_CONST_CONFIG_0_WGOFFSETCONSTID__SHIFT
        localidregid = (const_config_reg & mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__MASK) >> mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__SHIFT
        wge_cntl_reg = self.regs[mesa.REG_A6XX_SP_CS_WGE_CNTL]
        linearlocalidregid = (wge_cntl_reg & mesa.A6XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID__MASK) >> mesa.A6XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID__SHIFT
        run_asm(
            shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
            global_dims, group_dims, local_dims, kerneldim, wave_size,
            shared_memsize, pvt_memsize_peritem, pvt_mem_base,
            wgidconstid, wgsizeconstid, wgoffsetconstid, localidregid, linearlocalidregid, mergedregs
        )

    def submit_command(self, numcmds, cmdlist, cmdsize):
        cmd_struct = kgsl.struct_kgsl_command_object.from_address(cmdlist)
        if not cmd_struct.flags & kgsl.KGSL_CMDLIST_IB: raise NotImplementedError("Only KGSL_CMDLIST_IB is supported")
        self.queue.append(QCOMCommandObject(numcmds, cmdlist, cmdsize)) #kind of irrelevant
        inst_ptr = cmd_struct.gpuaddr
        end_ptr = inst_ptr + cmd_struct.size
        while inst_ptr < end_ptr:
            inst_ptr = self._decode_inst(inst_ptr)
        return len(self.queue) - 1 #TODO: probably change ret value