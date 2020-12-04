from matplotlib import pyplot as plt
plt.ion()

class LivePlot():
    def __init__(self,camera):
        self.camera = camera
        self.roi = [(500,1500),(1500,3500)]
        self.running = False
        self.plot_fps = 1
        self.figsize = (8,8)

    def init(self):
        from matplotlib import pyplot as plt
        self.fig = plt.figure(figsize=self.figsize)
        self.grid = plt.GridSpec(1, 1, hspace=0.025, wspace=0.025)
        self.ax1 = self.fig.add_subplot(self.grid[0,0])
        self.run_once()

    def run_once(self):
        self.ax1.cla()
        self.ax1.imshow(self.camera.queue.peek_last()[self.roi[0][0]:self.roi[0][1],self.roi[1][0]:self.roi[1][1]])
        self.ax1.set_title(f'Frame#: {self.camera.queue.global_rear} and exposure {round(self.camera.exposure_time/1000,2)}')
        plt.pause(0.01)
        plt.show()


    def run(self):
        while self.running:
            self.run_once()
            sleep(1/self.plot_fps)

    def start(self):
        from ubcs_auxiliary.threading import new_thread
        self.running = True
        new_thread(self.run)

    def stop(self):
        self.running = False

if __name__ == '__main__':
    from lcp_video.flir_camera_DL import FlirCamera
