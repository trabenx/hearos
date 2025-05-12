import tkinter as tk
from tkinter import messagebox


class AudioView:
    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("סיווג קטעי אודיו")

        # Listbox
        self.listbox = tk.Listbox(self.root, width=50, height=15, font=("Arial", 14))
        self.listbox.pack(padx=10, pady=10)

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.play_button = tk.Button(button_frame, text="נגן", width=10, command=self.controller.play_selected_clip)
        self.play_button.grid(row=0, column=0, padx=5)

        self.classify_violent_button = tk.Button(button_frame, text="סווג כאלים", width=15, command=lambda: self.controller.classify_selected_clip("violent"))
        self.classify_violent_button.grid(row=0, column=1, padx=5)

        self.classify_nonviolent_button = tk.Button(button_frame, text="סווג כלא אלים", width=15, command=lambda: self.controller.classify_selected_clip("non-violent"))
        self.classify_nonviolent_button.grid(row=0, column=2, padx=5)

        # Status label
        self.status_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.status_label.pack(pady=5)

    def start(self):
        self.root.mainloop()

    def populate_list(self, clips):
        self.listbox.delete(0, tk.END)
        for clip in clips:
            self.listbox.insert(tk.END, str(clip))

    def get_selected_index(self):
        selection = self.listbox.curselection()
        return selection[0] if selection else None

    def show_message(self, title, message):
        messagebox.showinfo(title, message)

    def show_error(self, title, message):
        messagebox.showerror(title, message)

    def set_status(self, message):
        self.status_label.config(text=message)
