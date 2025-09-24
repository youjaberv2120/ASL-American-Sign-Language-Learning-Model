import ffmpeg

def copy(input, output):
    ffmpeg.input(input).output(output)

def crop(inputFile, outputFile, width, height, x, y):
    ffmpeg.input(inputFile).crop(x, y, width, height).output(outputFile).run()

def shift(input, output, x_shift, y_shift):
    {
        ffmpeg.input(input)
        .filter('crop', f'iw-{abs(x_shift)}', f'ih-{abs(y_shift)}', abs(min(0, x_shift)), abs(min(0, y_shift)))
        .filter('pad', f'iw+{abs(x_shift)}', f'ih+{abs(y_shift)}', max(0, x_shift), max(0, y_shift))
        .output(output).run()
    }

def resize(input, output, scale):
    {
        ffmpeg.input(input)
        .filter('scale', f'iw*{scale}', '-1')
        .filter('pad', f'iw/{scale}', f'ih/{scale}', f'(ow-iw)/2', f'(oh-ih)/2')
        .output(output).run()
    }

def blur(input, output, blur_val):
    {
        ffmpeg.input(input)
        .filter('gblur', blur_val)
        .output(output).run()
    }

def resize_shift(input, output, x_shift, y_shift):
    {
        ffmpeg.input(input)
        .filter('crop', f'iw-{abs(x_shift)}', f'ih-{abs(y_shift)}', abs(min(0, x_shift)), abs(min(0, y_shift)))
        .output(output).run()
    }