def run_asm(
    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
    global_dims, group_dims, local_dims, kerneldim, thread_size,
    wgidconstid, wgsizeconstid, wgoffsetconstid, localidregid, linearlocalidregid
):
    print("reached run_asm")
    