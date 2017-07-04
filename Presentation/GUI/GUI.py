import tkinter as tk
from tkinter import font
from tkinter import filedialog
import pandas as pd
import numpy as np
import pickle

data: np.array

class PredictArea(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.text = tk.Text(self)

        self.button = tk.Button(self, text="Predict", command = self.predict, state = tk.DISABLED)
        
        self.button.pack(side="bottom", fill="both", expand=tk.YES)
        self.text.pack(fill="both", expand=tk.YES)
    

    def predict(self):
        review = self.text.get("1.0", tk.END)
        print(review)


class TrainArea(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        self.load_label = tk.Label(self, text="Load files: ").grid(row=0)
        self.load_button = tk.Button(self, text = "Load", command = self.load).grid(row=0, column = 1, sticky="E")
    
    def load(self):
        global x_train, y_train, vectorizer
        x_train_file = filedialog.askopenfile()
        x_train = pickle.load(x_train_file)

        y_train_file = filedialog.askopenfile()
        y_train = pickle.load(y_train_file)

        v_file = filedialog.askopenfile()
        vectorizer = pickle.load(v_file)

class MainApplication(tk.Frame):   
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.predict = PredictArea(self)
        self.train = TrainArea(self)
        
        self.train.pack(side="bottom", fill="both", expand=tk.YES)
        self.predict.pack(fill="both", expand=tk.YES)

def main():
    root = tk.Tk()
    root.title("Review Predictor (Author: Dominik Stanojević)")
    root.geometry("600x600")
    
    
    #default_font = font.nametofont("TkDefaultFont")
    #default_font.configure(size=20)
    #root.option_add("*Font", default_font)

    app = MainApplication(root).pack(fill="both", expand=tk.YES)
    root.mainloop()

if __name__ == '__main__':
    main()
