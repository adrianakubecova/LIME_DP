
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import io
import base64
from IPython.display import display, HTML
import re
import plotly.express as px
import matplotlib.patches as mpatches
scaler = MinMaxScaler()
import plotly.graph_objs as go


def fetch_abstract_by_doi(doi, df):
    if 'abstract_text' not in df.columns:
        raise ValueError("DataFrame does not contain 'abstract_text' column.")
    result = df[df['doi'] == doi]['abstract_text']
    if result.empty:
        raise ValueError(f"No record found for DOI: {doi}")

    return result.iloc[0]


def colorize_importance(importance, rank, decrease_amount=0.05, max_alpha=1.0):
    if importance > 0.5:
        alpha = max_alpha - rank * decrease_amount
        return f'rgba(255, 164, 56, {alpha})'
    else:
        alpha = max_alpha - rank * decrease_amount
        return f'rgba(135, 180, 250, {alpha})'


def colorize_importance_stacked_plot(importance, rank, decrease_amount=0.05, max_alpha=1.0):
    orange = (1.0, 140/255, 0)
    light_blue = (135/255, 206/255, 250/255)
    alpha = max_alpha - rank * decrease_amount
    alpha = max(0, alpha)
    return (orange + (alpha,)) if importance > 0 else (light_blue + (alpha,))


def colorize_importance_plot(importance, rank, total_ranks, decrease_amount=0.1, max_alpha=1.0, min_alpha=0.2):
    orange = (255, 140, 0) 
    light_blue = (135, 206, 250) 
    alpha = max_alpha - ((total_ranks - rank) * decrease_amount)
    alpha = min(max(alpha, min_alpha), max_alpha)
    orange_scaled = tuple([o / 255 for o in orange]) + (alpha,)
    light_blue_scaled = tuple([lb / 255 for lb in light_blue]) + (alpha,)

    return orange_scaled if importance > 0.5 else light_blue_scaled


def highlight_abstract_words(abstract, words_df):
    # Sort the words by importance so the most important words are highlighted first.
    words_df = words_df.sort_values(by='final_importance', ascending=False)

    for idx, (_, row) in enumerate(words_df.iterrows()):
        importance = row['final_importance']
        word = re.escape(row['word'])
        hue = colorize_importance(importance, rank=idx)
        text_color = 'white' if importance < 0.5 else 'black'
        
        def replacement_func(match):
            whole_word_match = rf'\b{match.group()}\b'
            if re.fullmatch(whole_word_match, match.group()):
                return f'<span style="background-color:{hue}; color:{text_color};">{match.group()}</span>'
            else:
                return match.group()
        
        pattern = rf'\b{word}\b'
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        abstract = compiled_pattern.sub(replacement_func, abstract)
    
    return abstract
    
def determine_importance(doi, df, global_importance_col, threshold, use_smer=False):
    subset = df[df['doi'] == doi].copy()
    if use_smer:
        subset['final_importance'] = np.where(
        abs(subset['importance_scaled'] - subset["score"]) < 0.2,
        subset["score"],
        subset['importance_scaled']
    )
    else:
        subset['final_importance'] = np.where(
            abs(subset['importance_norm'] - subset[global_importance_col]) < threshold,
            subset[global_importance_col],
            subset['importance_norm']
        )
    return subset


def plot_stacked_importance_chart(df, use_smer=False):
    sns.set_style("whitegrid")
    
    # Determine sorting column and calculate differences
    sort_by_col = 'smer_score' if use_smer else 'global_importance'
    df['base_importance'] = df[sort_by_col] - 0.5
    df['difference'] = df['local_importance'] - df[sort_by_col]
    df['local_shifted'] = df['local_importance'] - 0.5
    df['rank'] = df['base_importance'].abs().rank(method='first', ascending=False)
    df = df.sort_values('rank', ascending=True)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, len(df)))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    #fig.patch.set_alpha(0)
    ax.set_frame_on(False)
    df.set_index('word', inplace=True)

    ax.barh(df.index, df['base_importance'], left=0.5, color='grey', edgecolor='none', alpha=0.9)

    for idx, (word, row) in enumerate(df.iterrows()):
        rank = row['rank']
        color = colorize_importance_stacked_plot(row['difference'], rank)
        global_bar_value = row['base_importance']
        local_bar_value = row['local_shifted']
        global_bar_end = 0.5 + global_bar_value
        local_bar_end = 0.5 + local_bar_value
        
        if abs(global_bar_value) > abs(local_bar_value):
            ax.barh(word, global_bar_value, left=0.5, color='grey', edgecolor='none', alpha=0.9)
            ax.barh(word, local_bar_value, left=0.5, color=color, edgecolor='none', alpha=0.9)
        else:
            ax.barh(word, local_bar_value, left=0.5, color=color, edgecolor='none', alpha=0.9)
            ax.barh(word, global_bar_value, left=0.5, color='grey', edgecolor='none', alpha=0.9)

        
        if abs(global_bar_end - 0.5) > abs(local_bar_end - 0.5):
            label_x_position = global_bar_end
        else:
            label_x_position = local_bar_end

        if label_x_position < 0.5:
            text_align = 'right'
            label_padding = -0.001
        else:
            text_align = 'left'
            label_padding = 0.001

        # Place the word label
        ax.text(label_x_position + label_padding, idx - 0.2, word, va='center', ha=text_align, color='black', fontsize=18)
        # Place the global and local value labels
        text = f'Global: {global_bar_value + 0.5:.2f}, Local: {local_bar_value + 0.5:.2f}'
        ax.text(label_x_position + label_padding, idx + 0.2, text, va='center', ha=text_align, color='grey', fontsize=14)
    
    ax.set_title(f'Global {"SMER" if use_smer else "LIME"} importance vs local LIME importance', fontsize=16, fontweight='bold', color="black")
    global_patch = mpatches.Patch(color='grey', label=f"Global{' SMER' if use_smer else ' LIME'} importance")
    local_smaller_patch = mpatches.Patch(color='lightblue', label='Local importance smaller than global')
    local_bigger_patch = mpatches.Patch(color='orange', label='Local importance bigger than global')
    ax.legend(handles=[global_patch, local_smaller_patch, local_bigger_patch], loc='best', fontsize=13)
    max_importance = max(df['base_importance'].abs().max(), df['local_shifted'].abs().max())
    x_min = 0.5 - max_importance - 0.05
    x_max = 0.5 + max_importance + 0.05
    ax.set_xlim([x_min, x_max])
    ax.axvline(x=0.5, color='gray', linewidth=1, linestyle='-')
    plt.gca().invert_yaxis() 

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([0.5])
    ax.set_xticklabels(['0.5'], fontsize=12, color='grey')

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f'<img src="data:image/png;base64,{img_base64}" alt="Bar chart"/>'


def display_stacked_importance_chart(doi, df, use_smer=False, threshold=0.1, global_importance_col="global_avg_importance"):
    relevant_words_df = determine_importance(doi, df, global_importance_col, threshold, use_smer)
    if use_smer:
        relevant_words_df['distance_from_center'] = abs(relevant_words_df['score'] - 0.5)
    else:
        relevant_words_df['distance_from_center'] = abs(relevant_words_df['final_importance'] - 0.5)
    
    relevant_words_df = relevant_words_df.sort_values(by='distance_from_center', ascending=False)
    data_for_plot = pd.DataFrame([
        {
            "word": row['word'],
            "global_importance": row[f"{global_importance_col}_scaled"] if use_smer else row[global_importance_col],
            "local_importance": row["importance_scaled"] if use_smer else row["importance_norm"],
            # Add "smer_score" only if use_smer is True
            **({"smer_score": row['score']} if use_smer else {})
        }
        for _, row in relevant_words_df.iterrows()
    ])
    img_html_lime = plot_stacked_importance_chart(data_for_plot, use_smer)
    display(HTML(f"""
        <div style='display: flex; justify-content: center; align-items: center;'>
            <div style='flex: 2; padding: 10px; max-width: 1000px; word-wrap: break-word; overflow-y: auto;'>
                {img_html_lime}
            </div>
        </div>
    """))


def compute_word_scores_for_abstract(abstract, df, column='score', top_n=15, absolute=False):
    words = pd.Series(abstract.split()).unique()
    word_scores = []

    for word in words:
        scores = df[df['word'] == word][column].unique()
        if scores.size > 0:
            average_score = np.mean(scores)
            word_scores.append((word, average_score, abs(average_score - 0.5)))

    if absolute:
        sorted_words = sorted(word_scores, key=lambda x: abs(x[1] - 0.5))
        half_n = top_n // 2
        final_words = sorted_words[:half_n] + sorted_words[-half_n:]
    else:
        final_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_n]

    return pd.DataFrame(final_words, columns=['word', "final_importance", 'distance_from_center'])


def colorize_importance_bar_plots(importance, rank, total_ranks, max_alpha=1.0):
    orange = (255/255, 140/255, 0)
    light_blue = (135/255, 180/255, 250/255)
    alpha_scale = rank / total_ranks
    alpha = max_alpha * max((rank / total_ranks), 0.2)

    return (orange + (alpha,)) if importance > 0.5 else (light_blue + (alpha,))

    

def plot_importance_bar_chart(data, use_smer=False, include_smer_scores=False, threshold=0.1):
    sns.set_style("white")
    if use_smer:
        data = [item for item in data if np.isfinite(item['smer_score'])]
    else:
        data = [item for item in data if np.isfinite(item['final_importance'])]

    data = data[::-1]
    words = [item['word'] for item in data]
    importance = [item['smer_score'] if use_smer else item['final_importance'] for item in data]
    importance_shifted = [(value - 0.5) for value in importance]

    fig, ax = plt.subplots(figsize=(6, len(data) * 0.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    bar_colors = [colorize_importance_plot(value, idx, len(data) - 1) for idx, value in enumerate(importance)]
    bars = ax.barh(words, importance_shifted, color=bar_colors, edgecolor='none')
    for bar in bars:
        bar.set_linewidth(0.5)

    for index, value in enumerate(importance_shifted):
        word_label = words[index]
        local_metric = f'{value + 0.5:.3f}'
        global_metric = f'{data[index]["global_importance"]:.3f}'

        if use_smer:
            diff = abs(data[index]["local_importance"] - data[index]["smer_score"]) > 0.4
            importance_label = f'SMER: {importance[index]:.3f}' + (f' LIME: {data[index]["local_importance"]:.3f}' if diff else "")
        else:
            diff = abs(data[index]["local_importance"] - data[index]["global_importance"]) > threshold
            importance_label = f"{'LIME' if diff else 'GLIME'}: {local_metric}"
            if diff:
                importance_label += f" (GLIME: {global_metric})"

        if include_smer_scores and not use_smer:
            importance_label += f' SMER: {data[index]["smer_score"]:.3f}'
        # Display the word
        ha_position = 'right' if value < 0 else 'left'
        ax.text(value - 0.005 if value < 0 else value + 0.005, index + 0.2, word_label, color='black', va='center', ha=ha_position, fontsize=18, fontweight='bold')
        # Display the importance value
        label_color = 'red' if diff else 'grey'
        ax.text(value - 0.005 if value < 0 else value + 0.005, index - 0.2, importance_label, color=label_color, va='center', ha=ha_position, fontsize=14)


    max_shift_abs = abs(max(importance_shifted, key=abs))
    padding = 0.1
    ax.set_xlim(-max_shift_abs - padding, max_shift_abs + padding)
    ax.axvline(0, color='grey', lw=1)
    ax.set_xticks([0])
    ax.set_xticklabels(['0.5'], fontsize=12, color='grey')

    high_cited_handle = mpatches.Patch(color=colorize_importance_plot(1, 14, len(data)), label='High cited')
    low_cited_handle = mpatches.Patch(color=colorize_importance_plot(0, 14, len(data)), label='Low cited')

    ax.legend(handles=[high_cited_handle, low_cited_handle], loc='best', fontsize=12)
    title_text = ('Rescaled word importance' if not use_smer else 'Word importance')  + (' SMER' if use_smer else ' LIME') + (' with SMER Scores' if include_smer_scores and not use_smer else '')
    ax.set_title(title_text, fontsize=16, fontweight='bold', y=1.02, color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])

    # Save the figure into an image buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f'<img src="data:image/png;base64,{img_base64}" alt="Bar chart"/>'


    
def plot_importance_bar_charts_with_smer(data, use_smer=False):
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(6, len(data) * 0.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    importance_shifted_values = []
    for index, item in enumerate(data):
        word = item['word']
        importance = item['final_importance']
        importance_shifted = importance - 0.5
        importance_shifted_values.append(importance_shifted)
        # Determine color based on importance
        color = colorize_importance_bar_plots(importance, index, len(data) - 1)

        # Adjust bar position
        bar_position = 0.5 if importance_shifted > 0 else 0.5 + importance_shifted
        ax.barh(word, abs(importance_shifted), left=bar_position, color=color, edgecolor='none')

        # Adjust text positioning and padding
        label_x_padding = 0.005
        label_x_position = (0.5 + importance_shifted + label_x_padding) if importance_shifted > 0 else (0.5 + importance_shifted - label_x_padding)
        ha_position = 'left' if importance_shifted > 0 else 'right'

        # Display the word and importance value
        ax.text(label_x_position, index + 0.2, f"{word}", va='center', ha=ha_position, fontsize=18, color='black', fontweight='bold')
        ax.text(label_x_position, index - 0.2, f"{importance:.2f}", va='center', ha=ha_position, fontsize=18, color='grey')

    high_cited_handle = mpatches.Patch(color=colorize_importance_bar_plots(1, 14, len(data)), label='High cited')
    low_cited_handle = mpatches.Patch(color=colorize_importance_bar_plots(0, 14, len(data)), label='Low cited')
    ax.legend(handles=[high_cited_handle, low_cited_handle], loc='upper left', fontsize=12)

    max_shift_abs = abs(max(importance_shifted_values, key=abs))
    padding = 0.1
    ax.set_xlim(0.5 - max_shift_abs - padding, 0.5 + max_shift_abs + padding)
    ax.axvline(0.5, color='grey', lw=1)
    ax.set_xticks([0.5])
    ax.set_xticklabels(['0.5'], fontsize=12, color='grey')
    ax.set_title('Global word importance' + f"{' SMER' if use_smer else ' LIME'}", fontsize=16, fontweight='bold', color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f'<img src="data:image/png;base64,{img_base64}" alt="Bar chart"/>'


def display_plots_with_abstract(doi, df, global_importance_col="global_avg_importance_scaled", absolute=False):
    abstract = fetch_abstract_by_doi(doi, df)
    
    # LIME
    lime_words_df = compute_word_scores_for_abstract(abstract, df, column=global_importance_col, absolute=absolute)
    lime_words_df = lime_words_df.sort_values(by='final_importance')
    highlighted_abstract_lime = highlight_abstract_words(abstract, lime_words_df)
    img_html_lime = plot_importance_bar_charts_with_smer(lime_words_df.to_dict(orient='records'))

    # SMER
    #smer_words_df = compute_word_scores_for_abstract(abstract, df, column='score', absolute=absolute)
    #smer_words_df = smer_words_df.sort_values(by='final_importance')
    #highlighted_abstract_smer = highlight_abstract_words(abstract, smer_words_df)
    #img_html_smer = plot_importance_bar_charts_with_smer(smer_words_df.to_dict(orient='records'), use_smer=True)

    display(HTML(f"""
        <div style='display: flex; justify-content: center; align-items: center;'>
            <div style='flex: 2; padding: 10px; max-width: 600px; word-wrap: break-word; overflow-y: auto;'>
                {img_html_lime}
            </div>
            <div style='flex: 3; padding: 10px; padding-top: 20px; max-width: 800px; word-wrap: break-word; font-size: 1.2em;'>
                <h2>Explained instance</h2>
                <h4> DOI: {doi}</h4>
                {highlighted_abstract_lime}
            </div>
        </div>
    """))


def display_abstract_with_highlights(doi, df, global_importance_col="global_avg_importance", use_smer=False, threshold=0.1, include_smer_scores=False):
    abstract = fetch_abstract_by_doi(doi, df)
    relevant_words_df = determine_importance(doi, df, global_importance_col, threshold, use_smer)
    if use_smer:
        relevant_words_df['distance_from_center'] = abs(relevant_words_df['score'] - 0.5)
    else:
        relevant_words_df['distance_from_center'] = abs(relevant_words_df['final_importance'] - 0.5)
    
    relevant_words_df = relevant_words_df.sort_values(by='distance_from_center', ascending=False)
    highlighted_abstract = highlight_abstract_words(abstract, relevant_words_df)
    data_for_plot = [
    {
        "word": row['word'],
        "final_importance": row['final_importance'],
        "global_importance": row[f"{global_importance_col}_scaled"] if use_smer else row[global_importance_col],
        "local_importance": row["importance_scaled"] if use_smer else row["importance_norm"],
        # Add "smer_score" only if use_smer is True
        **({"smer_score": row['score']} if use_smer else {})
    }
    for _, row in relevant_words_df.iterrows()
]
    img_html = plot_importance_bar_chart(data_for_plot, use_smer, include_smer_scores, threshold)

    display(HTML(f"""
        <div style='display: flex; justify-content: center; align-items: center;'>
            <div style='flex: 2; padding: 10px; max-width: 700px; word-wrap: break-word; overflow-y: auto;'>
                {img_html}
            </div>
            <div style='flex: 3; padding: 10px; padding-top: 20px; max-width: 800px; word-wrap: break-word; font-size: 1.2em;'>
                <h2>Explained instance</h2>
                <h4> DOI: {doi}</h4>
                {highlighted_abstract}
            </div>
        </div>
    """))


def create_inconsistency_scatter_plot(df, global_importance_col):
    # Filter for inconsistent instances
    inconsistent_indices = df[df['is_inconsistent']].index
    df.loc[inconsistent_indices, 'importance_diff'] = (df.loc[inconsistent_indices, 'importance_norm'] - df.loc[inconsistent_indices, global_importance_col]).round(2)
    df.loc[inconsistent_indices, 'color'] = df.loc[inconsistent_indices, 'importance_diff'].apply(lambda x: 'red' if x >= 0 else 'blue')
    inconsistent_rows = df.loc[inconsistent_indices]

    # Create scatter plot
    fig = px.scatter(
        inconsistent_rows,
        x='importance_norm',
        y=global_importance_col,
        size='abstract_text_len',
        color='color',
        color_discrete_map={"red": "#ff7f0e", "blue": "#1f77b4"},
        hover_data={
            'importance_diff': True,
            'doi': True,
            'word': True,
            #'score': True,
            'importance_norm': True,
            global_importance_col: True,
            'abstract_text_len': False
        }
    )

    # Determine min and max values for axes
    min_value = min(inconsistent_rows['importance_norm'].min(), inconsistent_rows[global_importance_col].min())
    max_value = max(inconsistent_rows['importance_norm'].max(), inconsistent_rows[global_importance_col].max())

    # Add correlation line and boundary area
    fig.add_trace(
        go.Scatter(x=[min_value, max_value], y=[min_value, max_value], mode='lines', line=dict(color='darkgrey', width=1))
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value, max_value, min_value, min_value],
            y=[min_value - 0.1, max_value - 0.1, max_value + 0.1, min_value + 0.1, min_value - 0.1],
            fill='toself',
            fillcolor='rgba(220, 220, 220, 0.4)',
            mode='lines',
            line=dict(color='rgba(220, 220, 220, 0)'),
            name='Boundary area (+/- 0.1)'
        )
    )

    # Update hover template and layout
    fig.update_traces(
        hovertemplate='<br>'.join([
            'Importance: %{x:.2f}',
            'Global average LIME importance: %{y:.2f}',
            'Importance Diff: %{customdata[0]}',
            'DOI: %{customdata[1]}',
            'Word: %{customdata[2]}',
            #'Score: %{customdata[3]:.2f}',
            'Abstract Length: %{marker.size}'
        ])
    )

    fig.update_layout(
        title={
            'text': 'Local vs global average LIME importance',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        showlegend=False,
        xaxis_title='Local LIME importance',
        yaxis_title='Global average LIME importance',
        xaxis=dict(range=[min_value, max_value]),
        yaxis=dict(range=[min_value, max_value]),
        height=800,
        width=1200,
        plot_bgcolor='white'
    )

    fig.update_xaxes(
        title_text='Local LIME Importance',
        title_font=dict(size=18),
        showgrid=True,
        gridcolor='#f0f0f0',
        gridwidth=0.5
    )

    fig.update_yaxes(
        title_text='Global Average LIME Importance',
        title_font=dict(size=18),
        showgrid=True,
        gridcolor='#f0f0f0',
        gridwidth=0.5
    )

    fig.update_traces(
        hoverlabel=dict(
            namelength=-1,
            font_size=16
        )
    )

    fig.show()