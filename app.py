import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
GRAPH_PATH = os.path.join(STATIC_DIR, "graph.png")

os.makedirs(STATIC_DIR, exist_ok=True)

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

plt.style.use('dark_background')

furniture_types = {"yataq": 0, "skaf": 1, "masa": 2, "divan": 3, "kreslo": 4}
brands = {"ikea": 0, "dogtas": 1, "saloglu": 2, "premium": 3, "yerli": 4}

x = np.array([
    [0, 0, 8, 2], [1, 1, 9, 3], [2, 2, 7, 1],
    [3, 3, 10, 3], [0, 4, 6, 2], [1, 0, 8, 2],
    [4, 2, 5, 1], [3, 1, 8, 2], [2, 0, 9, 3]
])
y = np.array([800, 1500, 300, 3500, 500, 600, 250, 1200, 450])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
model = SGDRegressor(max_iter=10000, eta0=0.01, random_state=42)
model.fit(x_scaled, y)

@app.route('/', methods=["GET", "POST"])
def index():
    prediction = None
    timestamp = None
    error_msg = None

    if request.method == "POST":
        try:
            mebel_key = request.form.get("mebel_novu")
            marka_key = request.form.get("marka")

            if not mebel_key or not marka_key:
                error_msg = "Zəhmət olmasa bütün sahələri doldurun."
            else:
                mebel_val = furniture_types.get(mebel_key, 0)
                marka_val = brands.get(marka_key, 0)
                reytinq   = float(request.form.get("reytinq", 7))
                keyfiyyet = int(request.form.get("keyfiyyet", 2))

                new_data        = np.array([[mebel_val, marka_val, reytinq, keyfiyyet]])
                new_data_scaled = scaler.transform(new_data)
                predicted_price = model.predict(new_data_scaled)[0]
                prediction      = max(0, round(predicted_price, 2))

                fig, ax = plt.subplots(figsize=(8, 4.5))
                fig.patch.set_facecolor('#0d0d14')
                ax.set_facecolor('#0d0d14')

                ax.scatter(x[:, 0], y, color='#00e5ff', alpha=0.75,
                           s=80, label='Bazada olan qiymətlər', zorder=3)
                ax.scatter(mebel_val, prediction, marker="*", color="#d4af37",
                           s=350, label='Sizin Mebel', zorder=5)

                ax.set_xticks(list(furniture_types.values()))
                ax.set_xticklabels(list(furniture_types.keys()), color="#aaa", fontsize=11)
                ax.set_xlabel("Mebel Növü", color="#aaa", fontsize=12)
                ax.set_ylabel("Qiymət (AZN)", color="#aaa", fontsize=12)
                ax.set_title("Bazar Analizi Qrafiki", color="#d4af37", fontsize=14, pad=14)
                ax.tick_params(colors='#888')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2a2a3a')
                ax.grid(color='#1e1e2e', linestyle='--', linewidth=0.6)
                ax.legend(facecolor='#13131d', edgecolor='#333', labelcolor='#ccc')
                plt.tight_layout()

                plt.savefig(GRAPH_PATH, facecolor='#0d0d14', transparent=False, dpi=140)
                plt.close()

                timestamp = int(time.time())

        except ValueError as e:
            print(f"Dəyər xətası: {e}")
            error_msg = "Düzgün dəyər daxil edin."
        except Exception as e:
            print(f"Xəta baş verdi: {e}")
            error_msg = "Xəta baş verdi, yenidən cəhd edin."

    return render_template("index.html",
                           prediction=prediction,
                           timestamp=timestamp,
                           error_msg=error_msg)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
