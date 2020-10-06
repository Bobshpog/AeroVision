def slice_first_position_no_depth(input_photo):
    return input_photo[0,:, :, :3]

def remove_dc_photo(dc_photo, input_photo):
    return input_photo - dc_photo

def last_axis_to_first(input_photo):
    return input_photo.transpose((2,0,1))
def double_to_float(input_photo):
    return input_photo.astype('float32')
def mul_by1e7(x):
    return (1e7 * x).astype('float32')