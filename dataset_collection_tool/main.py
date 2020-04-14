import argparse
import random
import time
import tkinter as tk
import cv2


def gen_xy(master):
    x = random.uniform(0, master.winfo_screenwidth() - 150)
    y = random.uniform(0, master.winfo_screenheight() - 150)
    return (x, y)


class FullScreenApp:
    def __init__(self, master, annotation_file, **kwargs):
        self.annotation_file = annotation_file
        self.master = master
        self._geom = '200x200+0+0'
        self.target_x, self.target_y = gen_xy(master)
        self.look_at_me = tk.Label(master, text="Look\nat me!", bg="orange", font=("Arial Bold", 50))
        self.look_at_me.place(x=self.target_x, y=self.target_y)
        pad = 3
        master.geometry("{0}x{1}+0+0".format(master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        master.bind('<Escape>', self.toggle_geom)
        master.bind('<space>', self.iteration)

    def toggle_geom(self, event):
        geom = self.master.winfo_geometry()
        print(geom, self._geom)
        self.master.geometry(self._geom)
        self._geom = geom

    def move_target(self):
        master = self.master
        self.target_x, self.target_y = gen_xy(master)
        self.look_at_me.place(x=self.target_x, y=self.target_y, width=150, height=150)

    def iteration(self, event):
        timestamp = time.time_ns()
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        cv2.imwrite("img-{}.png".format(timestamp), frame)
        with open(self.annotation_file, "a") as f:
            f.write("{}\t{}\t{}\n".format(
                timestamp,
                self.target_x - self.master.winfo_screenwidth() / 2 + 75,
                self.target_y - self.master.winfo_screenheight() / 2 + 75)
            )
        self.move_target()


def get_args():
    parser = argparse.ArgumentParser(
        description='Dataset collection tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    output_params = parser.add_argument_group('Output')
    output_params.add_argument('--annotation-file',
                               type=str,
                               default="annotations.tsv",
                               metavar='FILENAME',
                               help='File to append annotations to')

    return parser.parse_args()


def main():
    args = get_args()
    root = tk.Tk()
    app = FullScreenApp(root, args.annotation_file)
    root.mainloop()


if __name__ == '__main__':
    main()
