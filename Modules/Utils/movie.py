
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


def movie_writer(tab, outfile, fps):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='flow movie', artist='Matplotlib',
                    comment='flow movie writer')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure()
    l = plt.imshow(tab[0])
    num_images = tab.shape[0]

    with writer.saving(fig, outfile, num_images):
        for i in range(num_images):
            curr_tab = tab[i]
            l.set_data(curr_tab)
            l.autoscale()
            writer.grab_frame()
    print('=> video writing in f done')


def flow_movie_writer(u, v, outfile, fps):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='flow movie', artist='Matplotlib',
                    comment='flow movie writer')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure()

    l = plt.imshow(u[0])
    num_images = u.shape[0]
    square_dim = u.shape[1]
    flow = np.concatenate((u.reshape(num_images, square_dim, square_dim, 1),
                           v.reshape(num_images, square_dim, square_dim, 1)), -1)

    with writer.saving(fig, outfile, num_images):
        for i in range(num_images):
            curr_tab = flow_to_image(flow[i])
            l.set_data(curr_tab)
            l.autoscale()
            writer.grab_frame()
    print('=> video writing in f done')

