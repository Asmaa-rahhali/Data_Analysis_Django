from django.shortcuts import render
import pandas as pd
import seaborn as sns
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import io
import base64

def home(request):
    data = None
    numeric_columns = []
    categorical_columns = []  
    statistics = None
    selected_column = None
    histogram = None
    heatmap = None
    boxplot = None
    scatterplot = None
    error_message = None
    countplot = None
    barplot = None
    piechart = None
    violinplot = None
    kdeplot = None
    heatmap_2cols = None
    if request.method == "POST":
        action = request.POST.get("action")  
        if "file" in request.FILES:
           
            file = request.FILES["file"]
            try:
                data = pd.read_excel(file)
                request.session['data'] = data.to_json(orient="split")  
            except Exception as e:
                error_message = f"Erreur lors de l'importation : {e}"
                data = None
        else:
            
            data_json = request.session.get('data')
            if data_json:
                data = pd.read_json(data_json, orient="split")
            else:
                data = None  

            if action == "statistics":
                selected_column = request.POST.get("selected_column")
                if selected_column and selected_column in data.columns:
                   
                    if "moyenne" in request.POST:
                        statistics = {"Moyenne": "{:.3f}".format(data[selected_column].mean())}
                    elif "mediane" in request.POST:
                        statistics = {"Médiane": data[selected_column].median()}
                    elif "mode" in request.POST:
                        mode_value = data[selected_column].mode()
                        statistics = {"Mode": mode_value.tolist()}
                    elif "variance" in request.POST:
                        statistics = {"Variance": "{:.3f}".format(data[selected_column].var())}
                    elif "ecart_type" in request.POST:
                        statistics = {"Écart Type": "{:.3f}".format(data[selected_column].std())}
                    elif "etendue" in request.POST:
                        etendue = data[selected_column].max() - data[selected_column].min()
                        statistics = {"Étendue": "{:.3f}".format(etendue)}

            elif action == "visualization":
                selected_column = request.POST.get("selected_column")
                if selected_column and selected_column in data.columns:
                    if "generate_histogram" in request.POST:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.hist(data[selected_column], bins=10, color="skyblue", edgecolor="black")
                        ax.set_title(f"Histogramme de {selected_column}")
                        ax.set_xlabel(selected_column)
                        ax.set_ylabel("Fréquence")
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        histogram = base64.b64encode(buf.read()).decode("utf-8")
                        buf.close()

                    if "generate_heatmap" in request.POST:
                        corr_data = data.select_dtypes(include=["number"]).corr()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                        ax.set_title("Heatmap des Corrélations")
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        heatmap = base64.b64encode(buf.read()).decode("utf-8")
                        buf.close()

                    if "generate_boxplot" in request.POST:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.boxplot(data=data, y=selected_column, ax=ax)
                        ax.set_title(f"Boxplot de {selected_column}")
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        boxplot = base64.b64encode(buf.read()).decode("utf-8")
                        buf.close()
                    if "generate_kdeplot" in request.POST:
                        try:
                            if selected_column in data.select_dtypes(include=["number"]).columns:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.kdeplot(data[selected_column].dropna(), fill=True, ax=ax)
                                ax.set_title(f"KDE Plot de {selected_column}")
                                buf = io.BytesIO()
                                plt.savefig(buf, format="png")
                                buf.seek(0)
                                kdeplot = base64.b64encode(buf.read()).decode("utf-8")
                                buf.close()
                                plt.close(fig)  
                            else:
                                error_message = f"La colonne '{selected_column}' ne peut pas être utilisée pour un KDE Plot."
                        except Exception as e:
                            error_message = f"Erreur lors de la génération du KDE Plot : {e}"

            elif action == "scatterplot":
                column_x = request.POST.get("column_x")
                column_y = request.POST.get("column_y")
                if column_x and column_y and column_x in data.columns and column_y in data.columns:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(x=data[column_x], y=data[column_y], ax=ax)
                    ax.set_title(f"Scatter Plot : {column_x} vs {column_y}")
                    ax.set_xlabel(column_x)
                    ax.set_ylabel(column_y)
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    scatterplot = base64.b64encode(buf.read()).decode("utf-8")
                    buf.close()
            elif action == "heatmap_2cols":
                column_x = request.POST.get("column_x")
                column_y = request.POST.get("column_y")

                if column_x and column_y and column_x in data.columns and column_y in data.columns:
                    subset = data[[column_x, column_y]]

                    corr_matrix = subset.corr()

                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    ax.set_title(f"Heatmap : {column_x} vs {column_y}")

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    heatmap_2cols = base64.b64encode(buf.read()).decode("utf-8")
                    buf.close()
                    plt.close(fig)
            elif action == "countplot":
                selected_categorical = request.POST.get("selected_categorical")
                if selected_categorical and selected_categorical in data.columns:
                    try:
                        if data[selected_categorical].dtype == 'object' or data[selected_categorical].dtype.name == 'category':
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.countplot(x=selected_categorical, data=data, ax=ax)  
                            ax.set_title(f"Countplot de {selected_categorical}")
                
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png")
                            buf.seek(0)
                            countplot = base64.b64encode(buf.read()).decode("utf-8")
                            buf.close()
                            plt.close(fig)  
                        else:
                            error_message = f"La colonne '{selected_categorical}' n'est pas catégorique."
                    except Exception as e:
                        error_message = f"Erreur lors de la génération du countplot : {e}"

            elif action == "barplot":
                selected_categorical = request.POST.get("selected_categorical")
                selected_numeric = request.POST.get("selected_numeric")
                if selected_categorical and selected_numeric and selected_categorical in data.columns and selected_numeric in data.columns:
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.barplot(x=selected_categorical, y=selected_numeric, data=data, ax=ax)
                        ax.set_title(f"Barplot de {selected_numeric} par {selected_categorical}")
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        barplot = base64.b64encode(buf.read()).decode("utf-8")
                        buf.close()
                        plt.close(fig)
                    except Exception as e:
                        error_message = f"Erreur lors de la génération du barplot : {e}"

            elif action == "piechart":
                selected_categorical = request.POST.get("selected_categorical")
                if selected_categorical and selected_categorical in data.columns:
                    category_counts = data[selected_categorical].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.set_title(f"Répartition des valeurs dans {selected_categorical}")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    piechart = base64.b64encode(buf.read()).decode("utf-8")
                    buf.close()

            elif action == "violinplot":
                selected_categorical = request.POST.get("selected_categorical")
                selected_numeric = request.POST.get("selected_numeric")
                if selected_categorical and selected_numeric and selected_categorical in data.columns and selected_numeric in data.columns:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.violinplot(x=data[selected_categorical], y=data[selected_numeric], ax=ax)
                    ax.set_title(f"Violin Plot de {selected_numeric} par {selected_categorical}")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    violinplot = base64.b64encode(buf.read()).decode("utf-8")
                    buf.close()
                else:
                    violinplot = None


    if data is not None:
        data_preview = pd.concat([data.head(3), data.tail(3)]).reset_index().to_dict(orient="records")
        numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
        columns = data.columns.tolist()
    else:
        data_preview = None
        numeric_columns = []
        categorical_columns = []
        columns = []

    return render(request, "home.html", {
        "data_preview": data_preview,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "statistics": statistics,
        "selected_column": selected_column,
        "columns": columns,
        "histogram": histogram,
        "heatmap": heatmap,
        "boxplot": boxplot,
        "scatterplot": scatterplot,
        "countplot": countplot,  
        "barplot": barplot,  
        "piechart": piechart,
        "violinplot": violinplot,  
        "kdeplot": kdeplot,  
        "error_message": error_message,
        "heatmap_2cols": heatmap_2cols, 
    })
def probabilites_view(request):
    result_bernoulli = None
    graph_bernoulli = None
    prob_success = 0  
    result_binomial = None
    chart_binomial = None
    result_loterie = None
    chart_loterie = None
    tickets_range = range(1, 11)
    result_poisson = None
    chart_poisson = None
    moyenne_par_heure = 3.5
    a=2
    b=5
    result_uniforme = None
    chart_uniforme = None
    result_exponentielle = None
    chart_exponentielle = None
    moyenne = 3  
    result_normale = None
    graph_normale = None

    mean = 75
    std_dev = 7
    if request.method == "POST":
        calculation_type = request.POST.get("calculation_type") 

        try:
            if calculation_type == "bernoulli":
                condition_type = request.POST.get("condition_type")
                condition_value = request.POST.get("condition_value")
                if condition_type == "equals":  
                    value = int(condition_value)
                    prob_success = 1 / 6
                    result_bernoulli = f"P(X = {value}) = 1/6 ≈ {round(prob_success, 4)}"
                elif condition_type == "greater":
                    value = int(condition_value) 
                    favorable = 6 - value
                    prob_success = favorable / 6
                    result_bernoulli = f"P(X > {value}) = {favorable}/6 ≈ {round(prob_success, 4)}"
                elif condition_type == "less":
                    value = int(condition_value)
                    favorable = value - 1
                    prob_success = favorable / 6
                    result_bernoulli = f"P(X < {value}) = {favorable}/6 ≈ {round(prob_success, 4)}"
                
                elif condition_type == "even":  
                    prob_success = 3 / 6
                    result_bernoulli = "P(X est pair) = 3/6 = 1/2 ≈ 0.5"

                elif condition_type == "odd":  
                    prob_success = 3 / 6
                    result_bernoulli = "P(X est impair) = 3/6 = 1/2 ≈ 0.5"

                graph_bernoulli = generate_bernoulli_graph(prob_success)

            elif calculation_type == "binomial":
                n = int(request.POST.get("n"))  
                x = int(request.POST.get("x"))  
                valeur = int(request.POST.get("valeur"))  

                p = 1 / 6  
                prob = binom.pmf(x, n, p)
                result_binomial = f"P= {prob:.4f}"
                x_values = list(range(n + 1))
                y_values = [binom.pmf(k, n, p) for k in x_values]

                plt.figure(figsize=(8, 5))
                plt.bar(x_values, y_values, color="skyblue", edgecolor="black")
                plt.title(f"Loi Binomiale \n Face recherchée : {valeur} ({n} lancers, {result_binomial})")

                plt.xlabel("Nombre de succès")
                plt.ylabel("Probabilité")
                plt.ylim(0, 1)  
                plt.yticks([i / 10 for i in range(11)])  
                plt.xticks(range(0, n + 1, 1))  

                plt.bar(x, binom.pmf(x, n, p), color="orange", edgecolor="black")
                plt.text(x, binom.pmf(x, n, p), f"{prob:.4f}", ha='center', va='bottom')

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                chart_binomial = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close()

            elif calculation_type == "loterie":
                n = int(request.POST.get("n"))  
                selection = int(request.POST.get("selection")) 

                if selection < 1 or selection > n:
                    raise ValueError(f"⚠️ Ticket sélectionné ({selection}) en dehors de l'intervalle [1, {n}].")

                prob = 1 / n
                result_loterie = f"P(Ticket {selection}) = 1/{n} = {prob:.4f}"

                labels = [f"Ticket {i}" for i in range(1, n + 1)]
                probabilities = [1 / n] * n

                plt.figure(figsize=(8, 5))
                plt.subplots_adjust(top=0.8)
                bars = plt.bar(labels, probabilities, color="skyblue", edgecolor="black")
                plt.bar(selection - 1, probabilities[selection - 1], color="orange", edgecolor="black")
                plt.title(f"Loi uniforme discontinue\nLoterie avec {n} tickets - Sélection {selection}", pad=20)

                plt.xlabel("Tickets")
                plt.ylabel("Probabilité")
                plt.ylim(0, 1.1)

                for i, bar in enumerate(bars):
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{probabilities[i]:.4f}", ha='center')

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                chart_loterie = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close()
            elif calculation_type == "poisson":
                x = int(request.POST.get("x"))
                heures = int(request.POST.get("heures"))

                moyenne_totale = moyenne_par_heure * heures

                prob_exacte = poisson.pmf(x, moyenne_totale)
                result_poisson = f"P(X = {x}) = {prob_exacte:.4f}"

                valeurs_x = list(range(0, x + 10))
                y_values = [poisson.pmf(k, moyenne_totale) for k in valeurs_x]

                plt.figure(figsize=(8, 5))
                plt.bar(valeurs_x, y_values, color="skyblue", edgecolor="black")
                plt.bar(x, poisson.pmf(x, moyenne_totale), color="orange", edgecolor="black") 
                plt.title(f"Loi de Poisson\n {x} appels en {heures} heures, P={prob_exacte:.4f}")
                plt.xlabel("Nombre d'événements")
                plt.ylabel("Probabilité")
                plt.xticks(range(0, x + 10))
                plt.yticks([i / 10 for i in range(11)])
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                chart_poisson = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close()

            elif calculation_type == "uniforme_continue":
                x = float(request.POST.get("x"))  
                condition = request.POST.get("condition")  
                if x < a or x > b:
                    raise ValueError(f"La valeur {x} est en dehors de l'intervalle [{a}, {b}].")

                f_x = 1 / (b - a)

                if condition == "=":  
                    prob = 0  
                    result_uniforme = f"P(X = {x}) = {prob:.4f}"

                elif condition == ">":  
                    prob = (b - x) / (b - a)
                    result_uniforme = f"P(X > {x}) = {prob:.4f}"

                elif condition == "<":  
                    prob = (x - a) / (b - a)
                    result_uniforme = f"P(X < {x}) = {prob:.4f}"

                valeurs_x = np.linspace(a - 1, b + 1, 100)
                y = [f_x if a <= v <= b else 0 for v in valeurs_x]

                plt.figure(figsize=(8, 5))
                plt.plot(valeurs_x, y, color='blue')  
                plt.fill_between(valeurs_x, y, 0, where=((valeurs_x >= a) & (valeurs_x <= b)), color='lightblue', alpha=0.5)

                if condition == "=":
                    plt.axvline(x, color='red', linestyle='--', label=f"x = {x}")
                elif condition == ">":
                    plt.fill_between(valeurs_x, y, 0, where=((valeurs_x >= x) & (valeurs_x <= b)), color='orange', alpha=0.6)
                elif condition == "<":
                    plt.fill_between(valeurs_x, y, 0, where=((valeurs_x >= a) & (valeurs_x <= x)), color='orange', alpha=0.6)

                plt.title(f"Loi Uniforme Continue [a={a}, b={b}] - Condition: {condition} {x}")
                plt.xlabel("Valeurs")
                plt.ylabel("Densité de probabilité")
                plt.legend()
                plt.grid(True)

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                chart_uniforme = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close()

            elif calculation_type == "exponentielle":
                x = float(request.POST.get("x"))
                condition = request.POST.get("condition")

                lambd = 1 / moyenne

                if condition == "greater":
                    result = 1 - expon.cdf(x, scale=1/lambd)
                    result_exponentielle = f"P(X > {x}) = {result:.4f}"

                elif condition == "less":
                    result = expon.cdf(x, scale=1/lambd)
                    result_exponentielle = f"P(X < {x}) = {result:.4f}"

                valeurs_x = np.linspace(0, x + 10, 100)
                y_values = expon.pdf(valeurs_x, scale=1/lambd)

                plt.figure(figsize=(8, 5))
                plt.plot(valeurs_x, y_values, color="blue", label="Densité de probabilité")
                plt.fill_between(valeurs_x, 0, y_values, where=(valeurs_x <= x) if condition == "less" else (valeurs_x >= x), color="skyblue", alpha=0.5)

                plt.axvline(x=x, color="red", linestyle="--", label=f"x = {x}")
                plt.title(f"Loi Exponentielle : λ = {lambd:.2f}")
                plt.xlabel("Durée")
                plt.ylabel("Densité de probabilité")
                plt.legend()

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                chart_exponentielle = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close()
            elif calculation_type == "normale":
                interval = request.POST.get('interval')

                if interval == '68_82':
                    prob = norm.cdf(82, mean, std_dev) - norm.cdf(68, mean, std_dev)
                elif interval == '61_89':
                    prob = norm.cdf(89, mean, std_dev) - norm.cdf(61, mean, std_dev)
                elif interval == '54_75':
                    prob = norm.cdf(75, mean, std_dev) - norm.cdf(54, mean, std_dev)
                elif interval == '96':
                    prob = 1 - norm.cdf(96, mean, std_dev)
                elif interval == '54':
                    prob = norm.cdf(54, mean, std_dev)
                else:
                    prob = 0

                result_normale = round(prob * 100, 3)

                x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
                y = norm.pdf(x, mean, std_dev)

                plt.figure(figsize=(8, 5))
                plt.plot(x, y, label='Loi Normale')
                plt.title('Distribution Normale avec µ=75 et σ=7')
                plt.xlabel('Valeurs')
                plt.ylabel('Densité de probabilité')

                if interval == '68_82':
                    plt.fill_between(x, y, where=(x >= 68) & (x <= 82), color='skyblue', alpha=0.5)
                elif interval == '61_89':
                    plt.fill_between(x, y, where=(x >= 61) & (x <= 89), color='skyblue', alpha=0.5)
                elif interval == '54_75':
                    plt.fill_between(x, y, where=(x >= 54) & (x <= 75), color='skyblue', alpha=0.5)
                elif interval == '96':
                    plt.fill_between(x, y, where=(x > 96), color='skyblue', alpha=0.5)
                elif interval == '54':
                    plt.fill_between(x, y, where=(x < 54), color='skyblue', alpha=0.5)

                plt.legend()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                graph_normale = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

        except Exception as e:
            if calculation_type == "bernoulli":
                result_bernoulli = f"⚠️ Erreur : {str(e)}"
            elif calculation_type == "binomial":
                result_binomial = f"⚠️ Erreur : {str(e)}"
            elif calculation_type == "loterie":
                result_loterie = f"⚠️ Erreur : {str(e)}"
            elif calculation_type == "poisson":
                result_poisson = f"⚠️ Erreur : {str(e)}"
            elif calculation_type == "uniforme_continue":
                result_uniforme = f"⚠️ Erreur : {str(e)}"
            elif calculation_type == "exponentielle":
                result_exponentielle = f"⚠️ Erreur : {str(e)}"

    if request.method == "GET":
        result_bernoulli = None
        graph_bernoulli = None
        result_binomial = None
        chart_binomial = None
        result_loterie = None
        chart_loterie = None
        result_poisson = None
        chart_poisson = None
        result_uniforme = None
        chart_uniforme = None
        result_exponentielle = None
        chart_exponentielle = None
        result_normale = None
        graph_normale = None
    return render(request, "probabilites.html", {
        "result_bernoulli": result_bernoulli,
        "graph_bernoulli": graph_bernoulli,
        "result_binomial": result_binomial,
        "chart_binomial": chart_binomial,
        "result_loterie": result_loterie,
        "chart_loterie": chart_loterie,
        "tickets_range": tickets_range,
        "result_poisson": result_poisson,
        "chart_poisson": chart_poisson,
        "result_uniforme": result_uniforme,
        "chart_uniforme": chart_uniforme,
        "a": a,
        "b": b,
        "moyenne": moyenne,
        "result_exponentielle": result_exponentielle,
        "chart_exponentielle": chart_exponentielle,
        'result_normale': result_normale,
        'graph_normale': graph_normale,
        
    })
def generate_bernoulli_graph(prob_success):
    """
    Génère un graphe pour la loi de Bernoulli avec des graduations fixes.
    """
    outcomes = ["Échec", "Succès"]
    probabilities = [1 - prob_success, prob_success]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(outcomes, probabilities, color=["red", "green"], edgecolor="black")

    ax.text(0, probabilities[0] + 0.02, f"P(Échec) = {round(probabilities[0], 4)}", ha='center')
    ax.text(1, probabilities[1] + 0.02, f"P(Succès) = {round(probabilities[1], 4)}", ha='center')

    ax.set_title(f"Loi de Bernoulli (p = {round(prob_success, 4)})")
    ax.set_xlabel("Résultat")
    ax.set_ylabel("Probabilité")
    ax.set_ylim(0, 1.1)  
    ax.set_yticks([i / 10 for i in range(11)])  

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    return graph