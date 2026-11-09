import math
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns

#Настройки 
VARIANT_NUMBER = 49
CSV_PATH = "traffic.csv"
K = 45 
MAX_LINE_POINTS = 1200 
MAX_SCATTER_POINTS = 12000
BG = "#f0f2f5"

METRICS = ["veh", "wait", "viol_r", "cong", "wait_ma", "cong_diff", "cong_norm", "cong_z"]
LEVELS = ["Все", "Низкий", "Средний", "Высокий"]
SLOTS = ["Ночь", "Утро", "День", "Вечер"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_RU = dict(zip(DAYS, ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]))
PALETTE = {"Низкий": "green", "Средний": "orange", "Высокий": "red"}
STEPS = {"авто": None, "1 мин": "1min", "15 мин": "15min", "1 час": "1h", "1 день": "1D"}

# Глобальное состояние
df_raw = None
df_work = None
fig = plt.Figure(figsize=(9, 5.5), dpi=100)
canvas = None
current_chart = "line"
_redraw_job = None

cam_var = metric_var = agg_var = level_var = step_var = None
smooth_var = wait_var = status_var = wait_label = None
last_rows = 0

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["path.simplify_threshold"] = 0.5
sns.set_theme(style="whitegrid")


#Загрузка
def load_data(path=CSV_PATH):
    """Чтение CSV с явными типами: память и groupby быстрее в разы."""
    global df_raw
    df_raw = pd.read_csv(path, dtype={"cam_id": "category", "veh": "float32",
                                      "wait": "float32", "cong": "float32",
                                      "viol_r": "float32"})
    df_raw["ts"] = pd.to_datetime(df_raw["ts"], unit="s")
    return df_raw


#Предобработка + FE
def clip_iqr(df, col, by="cam_id"):
    """Обрезка выбросов по IQR внутри каждой группы (этап 2, п. 3)."""
    g = df.groupby(by, observed=True)[col]
    q1 = g.transform("quantile", 0.25)
    q3 = g.transform("quantile", 0.75)
    iqr = q3 - q1
    return df[col].clip((q1 - 1.5 * iqr).clip(lower=0), q3 + 1.5 * iqr)


def preprocess_data():
    global df_work
    df = df_raw.copy() 

    df.loc[df["veh"] < 0, "veh"] = 0
    df.loc[df["wait"] < 0, "wait"] = 0
    df["viol_r"] = df["viol_r"].clip(0, 1)
    df = df.sort_values(["cam_id", "ts"], kind="mergesort").reset_index(drop=True)

    g = df.groupby("cam_id", observed=True)
    df["wait"] = clip_iqr(df, "wait").astype("float32")
    df["cong"] = clip_iqr(df, "cong").astype("float32")

    df["wait_ma"] = g["wait"].transform(lambda s: s.rolling(K, min_periods=1).mean())
    df["cong_diff"] = g["cong"].diff().fillna(0)

    st = g["cong"].agg(["min", "max"])

    cam_min = df["cam_id"].map(st["min"]).astype("float32")
    cam_rng = df["cam_id"].map(st["max"] - st["min"]).astype("float32").replace(0, np.nan)
    df["cong_norm"] = ((df["cong"] - cam_min) / cam_rng).fillna(0).astype("float32")
    z = df.groupby("cam_id", observed=True)["cong_norm"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0)
    df["cong_z"] = z.fillna(0).clip(-3, 3).astype("float32")

    dt = df["ts"].dt
    dow = dt.day_of_week if hasattr(dt, "day_of_week") else dt.dayofweek
    df["day_of_week"] = pd.Categorical.from_codes(dow.to_numpy(), DAYS, ordered=True)
    df["hour"] = dt.hour.astype("int8")
    df["time_slot"] = pd.Categorical(
        pd.cut(df["hour"], [-1, 5, 11, 17, 23], labels=SLOTS).astype(str), SLOTS, ordered=True)
    df["cong_level"] = pd.Categorical(
        pd.qcut(df["cong"], 3, labels=["Низкий", "Средний", "Высокий"],
                duplicates="drop").astype(str),
        ["Низкий", "Средний", "Высокий"], ordered=True)
    df["is_violation"] = (df["viol_r"] > df["viol_r"].median()).astype("int8")
    df["is_peak"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype("int8")

    df["cam_id"] = df["cam_id"].astype("category")
    df_work = df
    return df_work


#Фильтрация
def get_filtered():
    """Фильтры интерфейса: df.query() + df.loc[], затем .copy() — правило методички."""
    global last_rows
    df = df_work
    cam, level = cam_var.get(), level_var.get()
    if cam != "Все":
        df = df.query("cam_id == @cam")
    if level != "Все":
        df = df.query("cong_level == @level")
    threshold = wait_var.get()
    if threshold > 0:
        df = df.loc[df["wait"] >= threshold]
    last_rows = len(df)
    return df.copy()


def time_freq(n_points):
    """Шаг агрегации: выбранный пользователем либо авто (чтобы точек было ≤ MAX_LINE_POINTS)."""
    chosen = STEPS.get(step_var.get())
    if chosen:
        return chosen, step_var.get()
    minutes = max(1, math.ceil(n_points / MAX_LINE_POINTS))
    return f"{minutes}min", f"{minutes} мин"


#Графики
def clear_figure():
    fig.clear()  


def show_empty():
    clear_figure()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", fontsize=14)
    fig.tight_layout()
    canvas.draw_idle()


def plot_line():
    global current_chart
    current_chart = "line"
    df = get_filtered()
    if df.empty:
        return show_empty()

    metric, agg = metric_var.get(), agg_var.get()
    col, note = metric, ""
    if smooth_var.get():     
        df[col] = df.groupby("cam_id", observed=True)[metric].transform(
            lambda s: s.rolling(K, min_periods=1).mean())
        note = f", сглаживание k={K}"

    n_cam = df["cam_id"].nunique()
    freq, step_label = time_freq(len(df) / max(n_cam, 1))
    keys = [pd.Grouper(key="ts", freq=freq)] + (["cam_id"] if n_cam > 1 else [])
    agg_df = (df.groupby(keys, observed=True)[col].agg(agg)
                .reset_index().sort_values("ts"))

    clear_figure()
    ax = fig.add_subplot(111)
    if n_cam > 1:  
        wide = agg_df.pivot_table(index="ts", columns="cam_id", values=col,
                                   observed=True)
        long = wide.reset_index().melt(id_vars="ts", var_name="cam_id", value_name=col)
        sns.lineplot(data=long.dropna(), x="ts", y=col, hue="cam_id", ax=ax,
                     linewidth=1.0, errorbar=None, legend="brief")
        ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.6, title=None)
    else:
        sns.lineplot(data=agg_df, x="ts", y=col, ax=ax,
                     color="steelblue", linewidth=1.1, errorbar=None,
                     label=f"{agg}({col})")
        ax.legend(loc="upper left", fontsize=9)

    dc = (df.groupby(pd.Grouper(key="ts", freq=freq), observed=True)["cong_diff"]
            .mean().reset_index())
    ax2 = ax.twinx()
    sns.lineplot(data=dc, x="ts", y="cong_diff", ax=ax2, color="tomato",
                 linewidth=1.0, errorbar=None, label="Δcong", alpha=0.7)
    ax2.set_ylabel("Δcong", color="tomato")
    ax2.tick_params(axis="y", colors="tomato")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.6)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_title(f"{agg}({col}) во времени{note} (шаг {step_label})")
    ax.set_xlabel("Время")
    ax.set_ylabel(f"{agg}({col})")
    fig.tight_layout()
    canvas.draw_idle()


def plot_bar():
    global current_chart
    current_chart = "bar"
    df = get_filtered()
    if df.empty:
        return show_empty()

    metric, agg = metric_var.get(), agg_var.get()
    g = (df.groupby("cam_id", observed=True)
           .agg(value=(metric, agg), veh_max=("veh", "max"))
           .reset_index().sort_values("value", ascending=False))

    clear_figure()
    ax = fig.add_subplot(111)
    sns.barplot(data=g, x="cam_id", y="value", ax=ax, color="steelblue",
                errorbar=None, width=0.4)
    ax.set_ylabel(f"{agg}({metric})")
    if metric != "veh":
        ax2 = ax.twinx()                               
        ax2.bar(np.arange(len(g)) + 0.4, g["veh_max"], 0.4, color="tomato", alpha=0.85)
        ax2.set_ylabel("max(veh)", color="tomato")
        ax2.tick_params(axis="y", colors="tomato")
        ax2.set_ylim(0, g["veh_max"].max() * 1.15)

    ax.set_xlabel("Камера")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g["cam_id"], rotation=45, ha="right")
    ax.set_title(f"{agg}({metric}) и максимальный поток по камерам")
    fig.tight_layout()
    canvas.draw_idle()


def plot_scatter():
    global current_chart
    current_chart = "scatter"
    df = get_filtered()
    if df.empty:
        return show_empty()

    metric = metric_var.get()
    note = ""
    if len(df) > MAX_SCATTER_POINTS:
        df = df.sample(MAX_SCATTER_POINTS, random_state=42)
        note = f" (подвыборка {MAX_SCATTER_POINTS // 1000} тыс.)"

    clear_figure()
    ax = fig.add_subplot(111)
    sns.scatterplot(data=df, x="cong_z", y=metric, hue="cong_level", palette=PALETTE,
                    alpha=0.55, s=16, linewidths=0, ax=ax)
    for v in (-3, 3):
        ax.axvline(v, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("cong_z (per-cam, clip ±3)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs нормализованная загруженность{note}")
    fig.tight_layout()
    canvas.draw_idle()


def plot_heat_map():
    global current_chart
    current_chart = "heat"
    df = get_filtered()
    if df.empty:
        return show_empty()

    metric, agg = metric_var.get(), agg_var.get()
    pivot = df.pivot_table(index="day_of_week", columns="time_slot", values=metric,
                           aggfunc=agg, observed=True)
    if pivot.empty:
        return show_empty()
    pivot = pivot.reindex(index=[d for d in DAYS if d in pivot.index],
                          columns=[s for s in SLOTS if s in pivot.columns])

    clear_figure()
    ax = fig.add_subplot(111)
    if metric == "cong_z":
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    ax=ax, cbar_kws={"fraction": 0.035, "pad": 0.02})
    else:
        big = float(np.nanmax(np.abs(pivot.to_numpy())))
        sns.heatmap(pivot, annot=True, fmt=".0f" if big >= 100 else ".2f", cmap="viridis",
                    ax=ax, cbar_kws={"fraction": 0.035, "pad": 0.02})
    ax.set_yticklabels([DAY_RU[d] for d in pivot.index], rotation=0)
    ax.set_xlabel("Часть суток")
    ax.set_ylabel("День недели")
    ax.set_title(f"{agg}({metric}) по дням и частям суток")
    fig.tight_layout()
    canvas.draw_idle()


#Обновление / экспорт 
CHART_FUNCS = {"line": plot_line, "bar": plot_bar, "scatter": plot_scatter, "heat": plot_heat_map}


def redraw():
    """Единая точка перерисовки: вызывается из всех callback-функций."""
    global _redraw_job
    _redraw_job = None
    if df_work is None:
        return
    cam, level = cam_var.get(), level_var.get()
    suffix = ("" if cam == "Все" else f" — {cam}") + ("" if level == "Все" else f" / {level}")
    t0 = time.perf_counter()
    CHART_FUNCS[current_chart]()
    dt = (time.perf_counter() - t0) * 1000
    rows = f"строк: {last_rows:,} из {len(df_work):,}".replace(",", " ")
    wait_label.set(f"Порог wait ≥ {wait_var.get():.0f} с")
    status_var.set(f"{rows} | вариант {VARIANT_NUMBER} | перерисов: {dt:.0f} мс | "
                   f"{agg_var.get()}({metric_var.get()}){suffix}")


def schedule_redraw(*_):
    """Debounce через root.after() — вся работа строго в главном потоке."""
    global _redraw_job
    if _redraw_job is not None:
        root.after_cancel(_redraw_job)
    _redraw_job = root.after(120, redraw)


def refresh_data():
    """Перечитать файл и пересчитать признаки (кнопка «Обновить»)."""
    load_data()
    preprocess_data()
    redraw()


def export_plot():
    filepath = filedialog.asksaveasfilename(
        defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
    if not filepath:
        return
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    messagebox.showinfo("Экспорт", f"График сохранён:\n{filepath}")


#Интерфейс 
def build_ui(window):
    global canvas, cam_var, metric_var, agg_var, level_var, step_var
    global smooth_var, wait_var, status_var, wait_label

    window.title(f"Дашборд: вариант {VARIANT_NUMBER}")
    window.geometry("1300x850")
    window.configure(bg=BG)
    panel = tk.Frame(window, width=250, bg=BG)
    panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
    panel.pack_propagate(False)

    cam_var = tk.StringVar(value="Все")
    metric_var = tk.StringVar(value="wait_ma")
    agg_var = tk.StringVar(value="mean")
    level_var = tk.StringVar(value="Все")
    step_var = tk.StringVar(value="авто")
    smooth_var = tk.BooleanVar(value=False)
    wait_var = tk.DoubleVar(value=0.0)
    status_var = tk.StringVar(value="")
    wait_label = tk.StringVar(value="Порог wait ≥ 0 с")

    tk.Label(panel, text="Камера", bg=BG).pack(pady=(4, 2))
    combo = ttk.Combobox(panel, textvariable=cam_var, state="readonly",
                         values=["Все"] + sorted(df_work["cam_id"].astype(str).unique()))
    combo.pack(fill=tk.X)
    combo.bind("<<ComboboxSelected>>", schedule_redraw)

    tk.Label(panel, text="Метрика", bg=BG).pack(pady=(8, 2))
    combo = ttk.Combobox(panel, textvariable=metric_var, state="readonly", values=METRICS)
    combo.pack(fill=tk.X)
    combo.bind("<<ComboboxSelected>>", schedule_redraw)

    tk.Label(panel, text="Агрегация", bg=BG).pack(pady=(8, 2))
    for txt, val in (("Sum", "sum"), ("Mean", "mean"), ("Median", "median")):
        tk.Radiobutton(panel, text=txt, variable=agg_var, value=val, bg=BG,
                       command=schedule_redraw).pack(anchor="w")

    tk.Label(panel, text="Загруженность", bg=BG).pack(pady=(8, 2))
    combo = ttk.Combobox(panel, textvariable=level_var, state="readonly", values=LEVELS)
    combo.pack(fill=tk.X)
    combo.bind("<<ComboboxSelected>>", schedule_redraw)

    tk.Label(panel, text="Шаг агрегации", bg=BG).pack(pady=(8, 2))
    combo = ttk.Combobox(panel, textvariable=step_var, state="readonly",
                         values=list(STEPS))
    combo.pack(fill=tk.X)
    combo.bind("<<ComboboxSelected>>", schedule_redraw)

    tk.Checkbutton(panel, text=f"Сглаживание (rolling k={K})", variable=smooth_var,
                   bg=BG, command=schedule_redraw).pack(anchor="w", pady=(10, 0))

    tk.Label(panel, textvariable=wait_label, bg=BG).pack(pady=(8, 0))
    scale = ttk.Scale(panel, from_=0, to=60, variable=wait_var, command=schedule_redraw)
    scale.pack(fill=tk.X)

    tk.Label(panel, text="Тип графика", bg=BG).pack(pady=(12, 4))
    for txt, key in (("Линейный", "line"), ("Столбчатый", "bar"),
                     ("Точечный", "scatter"), ("Тепловая карта", "heat")):
        tk.Button(panel, text=txt, width=18,
                  command=lambda k=key: set_chart(k)).pack(fill=tk.X, pady=2)

    tk.Button(panel, text="Обновить", command=refresh_data).pack(fill=tk.X, pady=(14, 2))
    tk.Button(panel, text="Экспорт", command=export_plot).pack(fill=tk.X, pady=2)

    plot_frame = tk.Frame(window, bg="white", relief=tk.SUNKEN, bd=1)
    plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.get_tk_widget().bind("<Configure>", on_resize, add="+")

    tk.Label(window, textvariable=status_var, anchor="w", bg="#e9ecf1",
             padx=8).pack(side=tk.BOTTOM, fill=tk.X)


def set_chart(key):
    global current_chart
    current_chart = key
    redraw()


def on_resize(event):
    if event.width > 100 and event.height > 100:
        schedule_redraw()


def _on_error(exc, val, tb):
    import traceback
    traceback.print_exception(exc, val, tb)
    try:
        messagebox.showerror("Ошибка", f"{exc.__name__}: {val}")
    except Exception:
        pass


def main():
    global root
    load_data()
    preprocess_data()
    root = tk.Tk()
    root.report_callback_exception = _on_error
    build_ui(root)
    redraw()
    root.mainloop()


if __name__ == "__main__":
    main()
