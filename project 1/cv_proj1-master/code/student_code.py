import numpy as np
from functools import reduce

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  BUFFER_SIZE, HALF_BUFFER = get_buffer_sizes(filter)

  framed_image = frame_image(image, BUFFER_SIZE)

  rows, cols, _ = framed_image.shape
  temp_channels = []

  for channel in color_channels(framed_image):
      filtered_channel = np.empty((rows - BUFFER_SIZE * 2, cols - BUFFER_SIZE * 2))
      for i in range(BUFFER_SIZE, rows - BUFFER_SIZE):
          for j in range(BUFFER_SIZE, cols - BUFFER_SIZE):
              neighbors = []
              if is_1d_row(filter):
                  neighbors = channel[i, j - HALF_BUFFER:j + HALF_BUFFER + 1]
                  result = filter_neighborhood(filter, neighbors)
                  filtered_channel[i - BUFFER_SIZE][j - BUFFER_SIZE] = result
              elif is_1d_col(filter):
                  neighbors = channel[i - HALF_BUFFER:i + HALF_BUFFER + 1, j]
                  result = filter_neighborhood(filter, neighbors)
                  filtered_channel[i - BUFFER_SIZE][j - BUFFER_SIZE] = result
              else:
                  neighbors = channel[i - HALF_BUFFER:i + HALF_BUFFER + 1, j - HALF_BUFFER:j + HALF_BUFFER + 1]
                  result = filter_neighborhood(filter, neighbors)
                  filtered_channel[i - BUFFER_SIZE][j - BUFFER_SIZE] = result

      temp_channels.append(filtered_channel)

  filtered_image = stacker(temp_channels)
  return filtered_image

def is_1d_col(arr):
    return arr.shape[0] == 1

def is_1d_row(arr):
    return arr.shape[1] == 1

def get_buffer_sizes(filter):
    buffer_size = np.uint8(filter.shape[0])

    if is_1d_col(filter):
        buffer_size = np.uint8(filter.shape[1])

    half_buffer = np.uint8((buffer_size - 1) / 2)
    return buffer_size, half_buffer

def filter_neighborhood(filter, neighbors):
    rows, cols = filter.shape
    if rows == 1:
        return np.sum([np.dot(val, filter[0][index]) for index, val in enumerate(neighbors)])
    elif cols == 1:
        return np.sum([np.dot(val, filter[index]) for index, val in enumerate(neighbors)])
    else:
        return np.sum(np.multiply(filter, neighbors))

def frame_image(image, buffer_size):
    buff = np.int(buffer_size)
    channels = [np.pad(channel, ((buff,buff), (buff,buff)), 'symmetric') for channel in color_channels(image)]
    return stacker(channels)

def get_buffers(buffer_size, channel):
    row_count, col_count = channel.shape
    buffer_row = np.zeros((buffer_size, col_count))
    buffer_col = np.zeros((row_count + buffer_size * 2, buffer_size))
    return buffer_row, buffer_col

def color_channels(image):
    return [image[:,:,0], image[:,:,1], image[:,:,2]]

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  low_frequencies = my_imfilter(image1, filter)
  low_frequencies2 = my_imfilter(image2, filter)

  img2_channels = color_channels(image2)
  low_frequencies1_channels = color_channels(low_frequencies)
  low_frequencies2_channels = color_channels(low_frequencies2)

  high_frequencies = [img2_channel - low_frequencies2_channels[index] for index, img2_channel in enumerate(img2_channels)]
  hybrid = [high_frequencies[index] + low_channel1 for index, low_channel1 in enumerate(low_frequencies1_channels)]

  return low_frequencies, stacker(high_frequencies), clip_image(stacker(hybrid))

def clip_image(image):
    clipped_image = [np.clip(channel, 0, 1) for channel in color_channels(image)]
    return stacker(clipped_image)

def stacker(arr):
    return np.dstack((arr[0], arr[1], arr[2]))
