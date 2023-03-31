def update_namespace(kwargs):
    if 'T' in kwargs:
        kwargs['T'] = int(kwargs['T'])
    if 'T_fine' in kwargs:
        kwargs['T_fine'] = int(kwargs['T_fine'])
    globals().update(kwargs)