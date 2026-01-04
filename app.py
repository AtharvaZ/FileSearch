import os
# Suppress macOS Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import messagebox
import subprocess
import platform
from main import index_files, search_files
from database import Base, engine

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, bg_color, fg_color, hover_color, **kwargs):
        super().__init__(parent, bg=parent['bg'], highlightthickness=0, **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.hover_color = hover_color

        # Create rounded rectangle
        self.rect = self.create_rounded_rect(0, 0, kwargs.get('width', 120), kwargs.get('height', 40),
                                              radius=10, fill=bg_color, outline="")
        self.text = self.create_text(kwargs.get('width', 120)//2, kwargs.get('height', 40)//2,
                                     text=text, fill=fg_color, font=("SF Pro", 13, "bold"))

        self.bind("<Button-1>", lambda _: self.command())
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.config(cursor="hand2")

    def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                  x1+radius, y1,
                  x2-radius, y1,
                  x2-radius, y1,
                  x2, y1,
                  x2, y1+radius,
                  x2, y1+radius,
                  x2, y2-radius,
                  x2, y2-radius,
                  x2, y2,
                  x2-radius, y2,
                  x2-radius, y2,
                  x1+radius, y2,
                  x1+radius, y2,
                  x1, y2,
                  x1, y2-radius,
                  x1, y2-radius,
                  x1, y1+radius,
                  x1, y1+radius,
                  x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)

    def on_enter(self, _):
        self.itemconfig(self.rect, fill=self.hover_color)

    def on_leave(self, _):
        self.itemconfig(self.rect, fill=self.bg_color)

class FileSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File Search")
        self.root.geometry("750x500")

        # macOS blur effect - set window transparency and background
        self.root.attributes('-alpha', 0.915)  # 95% opacity for slight transparency
        self.root.configure(bg="#2b2b2b")

        # Try to enable macOS vibrancy/blur (works on macOS)
        try:
            self.root.attributes('-transparent', True)
            self.root.wm_attributes('-transparent', 'systemWindowBackgroundColor0')
        except:
            pass  # Not on macOS or not supported

        # Initialize database
        Base.metadata.create_all(engine)

        # Main container with semi-transparent background
        main_frame = tk.Frame(root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)

        # Title
        title_label = tk.Label(
            main_frame,
            text="File Search",
            font=("SF Pro Display", 24, "bold"),
            bg="#2b2b2b",
            fg="#ffffff"
        )
        title_label.pack(pady=(0, 20))

        # Search frame
        search_frame = tk.Frame(main_frame, bg="#2b2b2b")
        search_frame.pack(fill=tk.X, pady=(0, 20), padx=0)

        # Search entry container - Frame with background
        search_container = tk.Frame(search_frame, bg="#3a3a3a", height=35)
        search_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        search_container.pack_propagate(False)

        # Text variable
        self.search_var = tk.StringVar()
        self.search_timer = None  # Timer for auto-search

        # Create Entry widget
        self.search_entry = tk.Entry(
            search_container,
            textvariable=self.search_var,
            font=("SF Pro", 15),
            bg="#3a3a3a",
            fg="#ffffff",
            insertbackground="#007AFF",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0
        )
        self.search_entry.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Bind events for auto-search
        self.search_entry.bind("<Return>", lambda _: self.perform_search())
        self.search_var.trace_add('write', lambda *_: self.schedule_search())
        self.search_entry.focus()

        # Index button with rounded corners - matches search box height
        index_btn = RoundedButton(
            search_frame,
            text="Index",
            command=self.index_files,
            bg_color="#276d3e",
            fg_color="#ffffff",
            hover_color="#6a8a6a",
            width=76,
            height=30
        )
        index_btn.pack(side=tk.LEFT, padx=(10, 0))

        # Filter options frame
        filter_frame = tk.Frame(main_frame, bg="#2b2b2b")
        filter_frame.pack(fill=tk.X, pady=(10, 5), padx=6)

        # Checkbox to exclude code files
        self.exclude_code_var = tk.BooleanVar(value=False)
        exclude_code_check = tk.Checkbutton(
            filter_frame,
            text="Exclude code files (.py, .js, .cpp, etc.)",
            variable=self.exclude_code_var,
            font=("SF Pro", 11),
            bg="#2b2b2b",
            fg="#ffffff",
            selectcolor="#3a3a3a",
            activebackground="#2b2b2b",
            activeforeground="#ffffff",
            cursor="hand2",
            command=lambda: self.perform_search() if self.search_var.get().strip() else None
        )
        exclude_code_check.pack(side=tk.LEFT)

        # Results label
        self.results_label = tk.Label(
            main_frame,
            text="Enter a keyword to search",
            font=("SF Pro",11),
            bg="#2b2b2b",
            fg="#999999"
        )
        self.results_label.pack(anchor=tk.W, pady=(0, 0), padx=(6, 0))

        # Results container canvas for rounded corners
        results_canvas = tk.Canvas(main_frame, bg="#2b2b2b", highlightthickness=0)
        results_canvas.pack(fill=tk.BOTH, expand=True, padx=0)

        # Create rounded border (outer)
        results_border = self._create_rounded_rectangle(
            results_canvas, 0, 0, 760, 400, radius=16, fill="#5a5a5a", outline=""
        )

        # Create rounded background (inner)
        results_bg = self._create_rounded_rectangle(
            results_canvas, 2, 2, 758, 398, radius=15, fill="#3a3a3a", outline=""
        )

        # Results frame with scrollbar
        results_container = tk.Frame(results_canvas, bg="#3a3a3a")
        results_window = results_canvas.create_window(10, 10, anchor=tk.NW, window=results_container)

        # Scrollbar with modern styling
        scrollbar = tk.Scrollbar(results_container, bg="#3a3a3a", troughcolor="#3a3a3a",
                                activebackground="#666666", width=12)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 6), pady=4)

        # Results listbox
        self.results_listbox = tk.Listbox(
            results_container,
            font=("SF Pro", 11),
            bg="#3a3a3a",
            fg="#ffffff",
            selectbackground="#666666",
            selectforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            yscrollcommand=scrollbar.set,
            cursor="hand2",
            activestyle='none'
        )
        self.results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=4)
        scrollbar.config(command=self.results_listbox.yview)

        # Bind canvas resize
        results_canvas.bind("<Configure>", lambda e: self._resize_results(results_canvas, results_border, results_bg, results_window, e))

        # Bind click event
        self.results_listbox.bind("<Double-Button-1>", self.open_file)
        self.results_listbox.bind("<Return>", self.open_file)

        # Store results data
        self.current_results = []

    def schedule_search(self):
        """Schedule a search to execute after 0.36 seconds of inactivity"""
        # Cancel any existing timer
        if self.search_timer is not None:
            self.root.after_cancel(self.search_timer)

        # Schedule new search
        self.search_timer = self.root.after(365, self.perform_search)

    def perform_search(self):
        query = self.search_var.get().strip()

        if not query:
            # Clear results if search box is empty
            self.results_listbox.delete(0, tk.END)
            self.current_results = []
            self.results_label.config(text="Enter a keyword to search")
            return

        # Clear previous results
        self.results_listbox.delete(0, tk.END)
        self.current_results = []

        # Perform search using vector semantic search
        self.results_label.config(text=f"Searching for '{query}'...")
        self.root.update_idletasks()

        # Use the checkbox to determine if code files should be excluded
        exclude_code = self.exclude_code_var.get()
        results = search_files(query, k=15, exclude_code_files=exclude_code)

        if not results:
            self.results_label.config(text=f"No results found for '{query}'")
            self.results_listbox.insert(tk.END, "No files found matching your search.")
        else:
            self.results_label.config(text=f"Found {len(results)} file(s)")
            self.current_results = results

            for idx, result in enumerate(results):
                # Show file name with distance score
                display_text = f"  {result['file_name']}"
                self.results_listbox.insert(tk.END, display_text)

                # Alternate row colors for better readability
                if idx % 2 == 0:
                    self.results_listbox.itemconfig(idx, bg="#444444")

    def open_file(self, _=None):
        selection = self.results_listbox.curselection()

        if not selection:
            return

        index = selection[0]

        # Check if there are actual results
        if not self.current_results:
            return

        if index >= len(self.current_results):
            return

        file_path = self.current_results[index]['file_path']

        if not os.path.exists(file_path):
            messagebox.showerror("File Not Found", f"The file no longer exists:\n{file_path}")
            return

        # Open file based on platform
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', file_path])
            elif platform.system() == 'Windows':
                os.startfile(file_path)
            else:  # Linux
                subprocess.run(['xdg-open', file_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file:\n{str(e)}")

    def index_files(self):
        # Clear search box and results
        self.search_var.set("")
        self.results_listbox.delete(0, tk.END)
        self.current_results = []

        self.results_label.config(text="Indexing files...")
        self.root.update_idletasks()

        try:
            index_files()
            messagebox.showinfo("Success", "Files have been indexed successfully!")
            self.results_label.config(text="Indexing complete. Ready to search.")
        except Exception as e:
            messagebox.showerror("Error", f"Error indexing files:\n{str(e)}")
            self.results_label.config(text="Error during indexing")

    @staticmethod
    def _create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=25, **kwargs):
        """Create a rounded rectangle on a canvas"""
        points = [x1+radius, y1,
                  x1+radius, y1,
                  x2-radius, y1,
                  x2-radius, y1,
                  x2, y1,
                  x2, y1+radius,
                  x2, y1+radius,
                  x2, y2-radius,
                  x2, y2-radius,
                  x2, y2,
                  x2-radius, y2,
                  x2-radius, y2,
                  x1+radius, y2,
                  x1+radius, y2,
                  x1, y2,
                  x1, y2-radius,
                  x1, y2-radius,
                  x1, y1+radius,
                  x1, y1+radius,
                  x1, y1]
        return canvas.create_polygon(points, **kwargs, smooth=True)

    def _resize_search_box(self, canvas, bg_id, event):
        """Resize search box background when canvas resizes"""
        canvas.coords(bg_id, 0, 0, event.width, 48)
        # Update entry width
        for item in canvas.find_all():
            if canvas.type(item) == "window":
                canvas.itemconfig(item, width=event.width - 40)

    def _resize_results(self, canvas, border_id, bg_id, window_id, event):
        """Resize results container when canvas resizes"""
        canvas.coords(border_id, 0, 0, event.width, event.height)
        canvas.coords(bg_id, 2, 2, event.width - 2, event.height - 2)
        canvas.coords(window_id, 10, 10)
        # Update window size
        for item in canvas.find_all():
            if canvas.type(item) == "window":
                canvas.itemconfig(item, width=event.width - 20, height=event.height - 20)

if __name__ == "__main__":
    root = tk.Tk()
    app = FileSearchApp(root)
    root.mainloop()
