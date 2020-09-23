def slice_first_position_no_depth(input_photo):
    return input_photo[0, :, :, :3]

def remove_mean_photo(mean_photo, input_photo):
    return input_photo - mean_photo
