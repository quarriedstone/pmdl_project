import argparse
import random
import time
import tkinter as tk
import cv2


class FullScreenApp:
    def __init__(self, master, args):
        self.annotation_file = args.annotation_file
        self.inference_mode = args.inference_mode
        self.cap = cv2.VideoCapture(0)
        self.master = master
        self._geom = '200x200+0+0'
        self.look_at_me = tk.Label(master, text="Look\nat me!", bg="orange", font=("Arial Bold", 50))

        pad = 3
        master.geometry("{0}x{1}+0+0".format(master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        master.bind('<Escape>', self.toggle_geom)

        if self.inference_mode:
            master.bind('<space>', self.inference_iteration)
        else:
            self.random_move()
            master.bind('<space>', self.random_iteration)

    def toggle_geom(self, event):
        geom = self.master.winfo_geometry()
        print(geom, self._geom)
        self.master.geometry(self._geom)
        self._geom = geom

    def random_iteration(self, event):
        timestamp = time.time_ns()
        _, frame = self.cap.read()
        cv2.imwrite("img-{}.png".format(timestamp), frame)

        with open(self.annotation_file, "a") as f:
            f.write("{}\t{}\t{}\n".format(
                timestamp,
                self.target_x - self.master.winfo_screenwidth() / 2 + 75,
                self.target_y - self.master.winfo_screenheight() / 2 + 75)
            )

        self.random_move()

    def inference_iteration(self, event):
        _, frame = self.cap.read()
        self.infer_and_move(frame)

    def random_move(self):
        master = self.master
        x = random.uniform(0, master.winfo_screenwidth() - 150)
        y = random.uniform(0, master.winfo_screenheight() - 150)

        self.target_x, self.target_y = x, y
        self.look_at_me.place(x=self.target_x, y=self.target_y, width=150, height=150)

    def infer_and_move(self, frame):
        self.random_move()  # TODO add inference here
        # move the label by calling the following:
        # self.look_at_me.place(x=self.target_x, y=self.target_y, width=150, height=150)


def get_args():
    parser = argparse.ArgumentParser(
        description='Dataset collection tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--annotation-file',
                        type=str,
                        default="annotations.tsv",
                        metavar='FILENAME',
                        help='File to append annotations to')
    parser.add_argument('--inference-mode',
                        action='store_true',
                        help='enables inference mode')
    return parser.parse_args()


def main():
    args = get_args()
    root = tk.Tk()
    app = FullScreenApp(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
