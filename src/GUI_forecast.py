import tkinter as tk
import numpy as np
from xgb_model import forecast_from_gui
from preprocessing import load_and_prepare_data
from tkinter import ttk
from tkcalendar import Calendar
from datetime import datetime


class ForecastApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Tourism Forecast System")
        self.root.geometry("1100x820")
        self.root.configure(bg="#d8ecef")
        self.root.resizable(False, False)

        self.selected_date = datetime.today().date()

        self.build_ui()

    # =============================
    # MAIN UI
    # =============================
    def build_ui(self):

        self.card = tk.Frame(self.root, bg="white")
        self.card.place(relx=0.5, rely=0.5, anchor="center", width=980, height=760)

        tk.Label(
            self.card,
            text="ระบบพยากรณ์จำนวนนักท่องเที่ยว",
            font=("Arial", 24, "bold"),
            fg="black",
            bg="white"
        ).pack(pady=(40, 25))

        input_frame = tk.Frame(self.card, bg="white")
        input_frame.pack(pady=10)

        tk.Label(
            input_frame,
            text="วันเริ่มต้น:",
            font=("Arial", 15),
            fg="black",
            bg="white"
        ).grid(row=0, column=0, padx=15, pady=10)

        self.date_label = tk.Label(
            input_frame,
            text=str(self.selected_date),
            font=("Arial", 15, "bold"),
            fg="black",
            bg="#f1f1f1",
            width=12
        )
        self.date_label.grid(row=0, column=1)

        tk.Button(
            input_frame,
            text="เลือกวัน",
            font=("Arial", 13),
            bg="#1976d2",
            fg="black",
            relief="flat",
            width=10,
            command=self.open_calendar
        ).grid(row=0, column=2, padx=15)

        tk.Label(
            input_frame,
            text="พยากรณ์ล่วงหน้า:",
            font=("Arial", 15),
            fg="black",
            bg="white"
        ).grid(row=1, column=0, padx=15, pady=10)

        self.period_combo = ttk.Combobox(
            input_frame,
            values=["1 วัน", "7 วัน", "14 วัน", "30 วัน"],
            font=("Arial", 14),
            width=10,
            state="readonly"
        )
        self.period_combo.current(1)
        self.period_combo.grid(row=1, column=1)

        tk.Button(
            self.card,
            text="พยากรณ์",
            font=("Arial", 15, "bold"),
            bg="#e0e0e0",
            fg="black",
            relief="solid",
            width=18,
            command=self.run_forecast
        ).pack(pady=35)


        # หมายเหตุระบบ

        separator = tk.Frame(self.card, bg="#dddddd", height=1, width=600)
        separator.pack(pady=10)

        note_frame = tk.Frame(self.card, bg="#f9f9f9")
        note_frame.pack(pady=5, ipadx=10, ipady=8)

        tk.Label(
            note_frame,
            text="หมายเหตุ",
            font=("Arial", 11, "bold"),
            fg="#333333",
            bg="#f9f9f9"
        ).pack(pady=(0,5))

        tk.Label(
            note_frame,
            text=(
                "ระบบพยากรณ์นี้ใช้แบบจำลอง Machine Learning (XGBoost)\n"
                "ฝึกด้วยข้อมูลสถิติจำนวนนักท่องเที่ยวช่วงปี 2021–2025\n\n"
                "ผลลัพธ์เป็นค่าประมาณการจากแนวโน้มข้อมูลในอดีต\n"
                "อาจมีความคลาดเคลื่อนจากปัจจัยภายนอก"
            ),
            font=("Arial", 10),
            fg="#555555",
            bg="#f9f9f9",
            justify="center"
        ).pack()

        # ===== กรอบผลลัพธ์ =====
        self.result_frame = tk.Frame(
            self.card,
            bg="#fafafa",
            highlightbackground="black",
            highlightthickness=1
        )
        self.result_frame.place(relx=0.5, rely=0.78, anchor="center", width=780, height=250)

        self.result_text = tk.Text(
            self.result_frame,
            font=("Arial", 14),
            fg="black",
            bg="#fafafa",
            bd=0,
            wrap="word"
        )
        self.result_text.pack(expand=True, fill="both", padx=15, pady=15)
        self.result_text.config(state="disabled")

    # =============================
    # CALENDAR
    # =============================
    def open_calendar(self):

        popup = tk.Toplevel(self.root)
        popup.title("กำหนดวันที่เริ่มต้นการพยากรณ์")
        popup.geometry("400x480")
        popup.configure(bg="white")
        popup.grab_set()

        current = self.selected_date
        year = current.year
        month = current.month

        header = tk.Frame(popup, bg="white")
        header.pack(pady=10)

        tk.Label(
            popup,
            text="เลือกวันที่อ้างอิงสำหรับการพยากรณ์",
            font=("Arial", 18, "bold"),
            fg="black",
            bg="white"
        ).pack(pady=20)

        cal_frame = tk.Frame(popup, bg="white")
        cal_frame.pack(pady=10)

        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

        cal = Calendar(
            popup,
            selectmode="day",
            date_pattern="yyyy-mm-dd",
            font=("Arial", 14),

            showweeknumbers=False,

            # ให้แสดงวันเดือนอื่น
            showothermonthdays=True,

            foreground="black",
            headersforeground="black",
            normalforeground="black",
            weekendforeground="black",

            # ทำให้วันเดือนอื่นจาง
            othermonthforeground="#bfbfbf",
            othermonthbackground="white",

            selectforeground="red",
            selectbackground="white",

            bordercolor="black",
            background="white",

            height=6
        )

        cal.pack(pady=20)

        cal.tag_config(
            "selected",
            foreground="red",
            font=("Arial", 14, "bold")
        )

        tk.Button(
            popup,
            text="Confirm",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="black",
            width=18,
            relief="solid",
            command=lambda: self.confirm_date(cal, popup)
        ).pack(pady=35)

    def confirm_date(self, cal, popup):
        self.selected_date = datetime.strptime(
            cal.get_date(), "%Y-%m-%d"
        ).date()
        self.date_label.config(text=str(self.selected_date))
        popup.destroy()

    # =============================
    # FORECAST LOGIC
    # =============================
    def run_forecast(self):

        try:
            self.result_text.config(state="normal")
            self.result_text.delete("1.0", tk.END)

            days = int(self.period_combo.get().split()[0])

            ts = load_and_prepare_data("data/2021-2025.xlsx")

            dates, forecasts = forecast_from_gui(ts, self.selected_date, days)

            forecasts = [int(np.expm1(v)) for v in forecasts]

            smape = 11.68
            accuracy = 100 - smape

            self.result_text.insert(
                tk.END,
                f"รายงานผลการพยากรณ์จำนวนนักท่องเที่ยว\n"
                f"ระดับความแม่นยำของแบบจำลอง: {accuracy:.2f}%\n"
                f"วันที่อ้างอิงการพยากรณ์: {self.selected_date}\n\n"
            )

            for d, v in zip(dates, forecasts):

                lower = int(v * (1 - smape/100))
                upper = int(v * (1 + smape/100))

                self.result_text.insert(
                    tk.END,
                    f"{d.date()} → {v:,} คน  (ช่วง {lower:,} - {upper:,})\n"
                )

            if days > 1:

                first_val = forecasts[0]
                last_val = forecasts[-1]

                change_percent = ((last_val - first_val) / first_val) * 100
                direction = "เพิ่มขึ้น" if change_percent > 0 else "ลดลง"

                max_val = max(forecasts)
                max_date = dates[forecasts.index(max_val)]

                self.result_text.insert(tk.END, "\n")
                self.result_text.insert(
                    tk.END,
                    f"การเปลี่ยนแปลงตลอดช่วง {days} วัน: {direction} {abs(change_percent):.2f}%\n"
                    f"วันที่ที่มีนักท่องเที่ยวมากที่สุด: {max_date.date()} ({max_val:,} คน)\n"
                )

            self.result_text.config(state="disabled")

        except Exception as e:
            self.result_text.insert(tk.END, f"เกิดข้อผิดพลาด:\n{e}")
            self.result_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastApp(root)
    root.mainloop()