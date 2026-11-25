"""Stage12: preliminary report and all diagnostic figures from audited experiment rows."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from storage import ROOT

OUTPUT=ROOT/'results/preview'
matplotlib.rcParams['svg.hashsalt']='nirs-preview-v1'


def read(name):return json.loads((OUTPUT/name).read_text(encoding='utf-8'))


def number(value):return '—' if value is None else f'{value:.4f}'.replace('.',',')


def plot(fig,name):
    folder=OUTPUT/'plots';folder.mkdir(exist_ok=True)
    fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});plt.close(fig)


def names(row):
    return row['method'].replace('LogisticRegression','LR').replace('LinearSVC','SVC').replace('RandomForest','RF').replace('sql_context','context').replace('sql_only','SQL').replace('LLM ','')


def main():
    analysis=read('analysis.json');evidence=read('overfitting_evidence.json');diagnostics=read('diagnostic_results.json')
    data=analysis['common'];rows=[]
    for row in data:
        m=row['metrics'];s=m['successful_decision_metrics'];c=row['conservative_all_cases']['successful_decision_metrics']
        rows.append({'method':row['method'],'input_mode':row['input_mode'],'n':row['observations'],'groups':row['unique_groups'],
            'coverage':m['decision_coverage'],'F1':m['primary_f1'],'successful_only_F1':s['f1'],
            'conservative_F1':row['conservative_all_cases']['primary_f1'],'recall':s['attack_recall'],'FPR':s['false_positive_rate'],
            'TN':c['collapsed_binary_counts']['tn'],'FP':c['collapsed_binary_counts']['fp'],
            'FN':c['collapsed_binary_counts']['fn'],'TP':c['collapsed_binary_counts']['tp'],
            'template_weighted_F1':row['equal_group_weighted']['primary_f1'],'run_id':row['run_id']})
    pd.DataFrame(rows).to_csv(OUTPUT/'summary_table.csv',index=False,encoding='utf-8')
    pd.DataFrame(analysis['error_cases']).to_csv(OUTPUT/'error_cases.csv',index=False,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    labels=[names(row) for row in data]
    for ax,key,title in zip(axes,('F1','FPR'),('F1: общий предварительный срез, n=30','FPR на 15 штатных случаях')):
        values=[row[key] if row[key] is not None else np.nan for row in rows]
        ax.bar(range(len(rows)),values,color='#0072B2');ax.set_ylim(0,1.05);ax.set_title(title)
        for index,value in enumerate(values):
            if not np.isfinite(value):ax.text(index,.035,'нет оценки',rotation=90,ha='center',fontsize=7)
        ax.set_xticks(range(len(rows)),labels,rotation=65,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
    plot(fig,'comparison')
    for kind in ('learning','complexity','compactness'):
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
        for ax,method in zip(axes,('LogisticRegression','LinearSVC','RandomForest')):
            base=next(row for row in diagnostics if row['method']==method and row['kind']=='learning' and row['level']=='1.0')
            selected=[row for row in diagnostics if row['method']==method and row['kind']==kind]
            if kind!='learning':selected=[base]+selected
            if kind=='learning':
                selected.sort(key=lambda row:row['fit_rows']);x=[row['fit_rows'] for row in selected];ax.set_xlabel('Обучающих наблюдений')
            elif kind=='compactness':
                selected.sort(key=lambda row:row['columns']);x=[row['columns'] for row in selected];ax.set_xlabel('Фактическое число признаков')
            else:
                x=[0,1];ax.set_xticks(x,['Базовый','Сильное ограничение']);ax.tick_params(axis='x',labelrotation=15)
            ax.plot(x,[row['train_metrics']['primary_f1'] for row in selected],'o-',label='Train')
            ax.plot(x,[row['validation_metrics']['primary_f1'] for row in selected],'s-',label='Validation')
            ax.set_title(method);ax.set_ylim(0,1.03);ax.grid(alpha=.2)
        axes[0].set_ylabel('Binary F1');axes[-1].legend();plot(fig,kind)
    transfer=evidence['transfer'];methods=sorted({row['method'] for row in transfer})
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,direction in zip(axes,('transfer-web_app-to-wordpress','transfer-wordpress-to-web_app')):
        selected=[row for row in transfer if row['direction']==direction]
        ax.bar(range(len(selected)),[row['metrics']['successful_decision_metrics']['false_positive_rate'] for row in selected],color='#D55E00')
        ax.set_xticks(range(len(selected)),[row['method'].replace('Historical ','Old ').replace('TF-IDF + ','') for row in selected],rotation=45,ha='right',fontsize=8)
        ax.set_title(direction.replace('transfer-',''));ax.set_ylim(0,1.05);ax.set_ylabel('FPR');ax.grid(axis='y',alpha=.2)
    plot(fig,'transfer')
    sensitivity=evidence['neutral_name_sensitivity'];fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(range(len(sensitivity)),[row['alarm_flips']/row['probes'] for row in sensitivity],color='#009E73')
    ax.set_xticks(range(len(sensitivity)),[row['method'].replace('Historical ','Old ').replace('TF-IDF + ','') for row in sensitivity],rotation=30,ha='right')
    ax.set_ylabel('Доля смен сигнала');ax.set_ylim(0,1);ax.set_title('Чувствительность к названиям полей: 5 нейтральных проб');plot(fig,'name_sensitivity')
    robust=[row for row in read('ml_results.json') if row['phase']=='robustness']
    fig,ax=plt.subplots(figsize=(10,4));labels=[row['method']+' '+row['input_mode']+' '+row['transform'] for row in robust]
    ax.bar(range(len(robust)),[row['metrics']['primary_f1'] for row in robust],color='#CC79A7')
    ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right',fontsize=7);ax.set_ylim(0,1.05);ax.set_ylabel('F1');ax.set_title('Устойчивость: 120 контролируемых сценариев');plot(fig,'robustness')
    fig,axes=plt.subplots(3,6,figsize=(14,7));axes=list(axes.flat)
    for ax,row in zip(axes,data):
        matrix=np.asarray(row['conservative_all_cases']['successful_decision_metrics']['confusion_matrix'])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=15)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(int(matrix[i,j])),ha='center',va='center')
        ax.set_title(names(row),fontsize=7);ax.set_xticks([0,1]);ax.set_yticks([0,1])
    for ax in axes[len(data):]:ax.axis('off')
    fig.suptitle('Матрицы ошибок; при отказе используется консервативный учёт');plot(fig,'confusion_matrices')
    # Conditional intervals stay next to point estimates; they are not population guarantees.
    fig,ax=plt.subplots(figsize=(12,5))
    centers=[];lower=[];upper=[];labels=[]
    for row in data:
        interval=row['group_bootstrap_F1_95'];value=row['conservative_all_cases']['primary_f1']
        if interval is not None and value is not None:
            centers.append(value);lower.append(max(0,value-interval[0]));upper.append(max(0,interval[1]-value));labels.append(names(row))
    ax.errorbar(range(len(labels)),centers,yerr=[lower,upper],fmt='o',capsize=3)
    ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right',fontsize=8);ax.set_ylim(0,1.05)
    ax.set_ylabel('F1 и 95% групповой bootstrap');ax.set_title('Неопределённость малого предварительного среза');plot(fig,'uncertainty')
    legacy=read('legacy_results.json') if (OUTPUT/'legacy_results.json').exists() else None
    if legacy:
        fig,ax=plt.subplots(figsize=(8,4));grouped=legacy['grouped_holdout']
        ax.bar(range(len(grouped)),[row['metrics']['primary_f1'] for row in grouped])
        ax.set_xticks(range(len(grouped)),[row['method'] for row in grouped]);ax.set_ylabel('Macro-F1, 5 исходных классов')
        ax.set_ylim(0,1);ax.set_title('NIRS: групповой holdout, 6579 строк');plot(fig,'legacy_grouped')
    timing=read('timing_and_cost.json')
    fig,ax=plt.subplots(figsize=(12,5));time_rows=[row for row in timing['methods'] if row['seconds_per_observation_amortized'] is not None]
    ax.bar(range(len(time_rows)),[max(row['seconds_per_observation_amortized'],1e-6) for row in time_rows]);ax.set_yscale('log')
    ax.set_xticks(range(len(time_rows)),[row['method'].replace('LogisticRegression','LR').replace('RandomForest','RF').replace('LinearSVC','SVC') for row in time_rows],rotation=65,ha='right',fontsize=8)
    ax.set_ylabel('Секунд на наблюдение, логарифмическая шкала');ax.set_title('Удельное время; LLM — амортизированное время API-батча');plot(fig,'latency')
    threats=('sqli','privilege','exfiltration');fig,ax=plt.subplots(figsize=(9,7))
    values=np.full((len(data),3),np.nan)
    for i,row in enumerate(data):
        for j,threat in enumerate(threats):
            matched=next((item for item in analysis['slices'] if item['run_id']==row['run_id'] and item['slice_kind']=='threat' and item['slice']==threat),None)
            if matched:values[i,j]=matched['conservative_all_cases']['successful_decision_metrics']['attack_recall']
    image=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');fig.colorbar(image,ax=ax,label='Recall с консервативным учётом отказов')
    ax.set_xticks(range(3),threats);ax.set_yticks(range(len(data)),[names(row) for row in data],fontsize=8)
    ax.set_title('Полнота по угрозам: по 5 атак на ячейку');plot(fig,'threat_heatmap')
    explanation=read('linear_explanation.json') if (OUTPUT/'linear_explanation.json').exists() else None
    if explanation:
        contributions=explanation['top_changed_feature_contributions'][:12]
        fig,ax=plt.subplots(figsize=(9,5));ax.barh(range(len(contributions)),[row['contribution'] for row in contributions])
        ax.set_yticks(range(len(contributions)),[repr(row['feature']) for row in contributions]);ax.invert_yaxis()
        ax.set_title('Исходный SVC: вклад изменившихся n-грамм');ax.set_xlabel('Вклад в изменение разности оценок классов');plot(fig,'ngram_contributions')
    old_svc=next(row for row in transfer if 'Historical' in row['method'] and 'LinearSVC' in row['method'] and row['direction']=='transfer-wordpress-to-web_app')
    old_fpr=old_svc['metrics']['successful_decision_metrics']['false_positive_rate']
    old_names=next(row for row in sensitivity if 'Historical' in row['method'] and 'LinearSVC' in row['method'])
    lines=['# Предварительные результаты этапов 10–12','',
        '**Статус: сокращённое исследование. Полный прогон перенесён в этап 13.** Общее сравнение содержит 30 наблюдений '
        '(по 10 SQLi, повышения привилегий и доступа к данным; 5 атак и 5 штатных в каждой группе). '
        'Это разведочный срез, после просмотра он считается открытым. Повтор на нём не является новым слепым тестом.', '',
        '## Ответ на замечание о переобучении','',
        f'Проблема исходной модели воспроизводится: у архивного LinearSVC на 150 штатных запросах web_app '
        f'FPR составила {number(old_fpr)}; замена названий нейтральных полей изменила бинарный сигнал в {old_names["alarm_flips"]} из {old_names["probes"]} проб. '
        'Числа относятся к текущему зафиксированному срезу и отличаются от чисел ревьюера по объёму и составу.', '',
        'У новых маскированных профилей в этих пяти пробах смен сигнала нет. Это ограниченный контроль имён, '
        'а не доказательство устранения переобучения. На сокращённом переносе web_app → WordPress новые фиксированные '
        'профили дают высокую FPR, обратное направление заметно лучше. Ошибки могут отражать смену домена, '
        'ограниченный train и фиксированный порог, а не только классическое переобучение.', '',
        '**Вывод: исходная зависимость от имён подтверждена; считать проблему переноса решённой пока нельзя.** '
        'Основные ML-профили обучены на 2720 примерах; внутренние кривые используют отдельные train/validation группы. '
        'Параметры фиксированы, большой подбор прерван по решению пользователя. Данные теста не использовались '
        'для улучшения этих профилей после получения результатов.', '',
        '| Модель | Train F1 | Validation F1 | Разрыв |','|---|---:|---:|---:|']
    for row in evidence['train_validation_gaps']:lines.append(f'| {row["method"]} | {number(row["train_F1"])} | {number(row["validation_F1"])} | {number(row["gap"])} |')
    lines+=['','![Перенос](plots/transfer.png)','', '![Зависимость от имён](plots/name_sensitivity.png)','',
        '## Общее сравнение','', '| Метод | Покрытие | F1 | F1 с отказами как ошибками | Recall | FPR |','|---|---:|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+' | '.join([row['method'],number(row['coverage']),number(row['F1']),number(row['conservative_F1']),number(row['recall']),number(row['FPR'])])+' |')
    lines+=['', 'SQL-only и контекстные профили обозначены отдельно. Все строки используют одни и те же 30 ID. '
        'Часть ответов GPT-OSS сохранена из ранее выполненных батчей большого прогона; их происхождение и размеры '
        'батчей сохранены. Это не повторное получение тех же ответов в новых батчах.', '',
        'F1 отсутствует при неполном покрытии. Отказы дополнительно считаются ошибкой соответствующего истинного '
        'класса в консервативной оценке; знаменатель не уменьшается. Результаты по угрозам, источникам, диалектам, '
        'контексту, длине и обфускации находятся в `analysis.json`. Метрика с одинаковым весом связанных шаблонов '
        'показывается отдельно от обычной метрики по строкам.', '',
        '![Сравнение](plots/comparison.png)','', '![Матрицы ошибок](plots/confusion_matrices.png)','',
        '![Интервалы](plots/uncertainty.png)','',
        '![Полнота по угрозам](plots/threat_heatmap.png)','',
        'Даже 0 ложных тревог из 15 штатных случаев не подтверждает низкую эксплуатационную FPR: '
        'описательная верхняя граница Wilson 95% составляет около 20%. Групповой bootstrap учитывает имеющиеся '
        'связанные компоненты, но малое число групп ограничивает выводы. Построчные интервалы Wilson требуют '
        'независимости строк и приводятся только как описательный ориентир.', '',
        '## Диагностические графики','', '![Обучение](plots/learning.png)','', '![Сложность](plots/complexity.png)','',
        '![Компактность](plots/compactness.png)','', '![Устойчивость](plots/robustness.png)','',
        '![Удельное время](plots/latency.png)','',
        'Кривые построены по небольшому числу фиксированных профилей. Изменение числа признаков не обязательно '
        'так же уменьшает сериализованный пайплайн: словарь до отбора может сохраняться в TF-IDF. '
        'Фактические размеры моделей и число столбцов находятся в `diagnostic_results.json`.', '',
        f'Для сокращённого LLM-профиля выполнено {timing["new_preview_API_attempts"]} новых API-вызовов, '
        f'переиспользовано {timing["reused_decisions"]} готовых решений. До сокращения большого прогона было выполнено '
        f'{timing["previous_interrupted_full_API_attempts"]} обращений; они сохранены и не скрыты из истории расхода. '
        'Токены, задержки батчей и полное время с ожиданиями находятся в `timing_and_cost.json`. '
        'Цена аккаунта не установлена, поэтому денежная стоимость не выдумывается.', '',
        '![Линейные вклады](plots/ngram_contributions.png)','',
        'Для архивного SVC проверено точное восстановление разности оценок классов из весов активных n-грамм и смещения. '
        'В `linear_explanation.json` сохранены исходные пары запросов, оценки и вклады признаков; '
        'объяснение решения не является доказательством вредоносного намерения.', '',
        '## Исторический контроль','',
        'Архивные модели и их метрики сохранены отдельно от новых бинарных моделей. Для проверки переноса '
        'сигнал исходной пяти-классовой модели определяется как «любая метка, кроме legit». Это преобразование '
        'решения, а не объявление старых исходных меток проверенной бинарной разметкой. '
        'Групповой NIRS holdout и исходные строковые CV/holdout имеют разные протоколы и выбор признаков; '
        'их нельзя интерпретировать как чистый эффект единственного изменения.', '',
        '![Исторический групповой контроль](plots/legacy_grouped.png)','',
        '## Воспроизведение','',
        '```powershell','.\\.venv\\Scripts\\python.exe -B src/evaluate.py --manifest results/preview/run_manifest.json',
        '.\\.venv\\Scripts\\python.exe -B src/compare.py --manifest results/preview/run_manifest.json','```','',
        'Эти команды читают зарегистрированные прогнозы PostgreSQL, пересчитывают результаты и создают графики '
        'без новых LLM-вызовов. Код сохранения проверяет закреплённые аннотации и test/validation принадлежность. '
        'В `audit_report.json` сверены метрики независимым расчётом sklearn и контрольные суммы связанных артефактов.', '',
        '## Дальше: этап 13','',
        'После отдельного запуска полного профиля завершить все фолды, полный общий LLM-прогон и контекстную '
        'абляцию; затем заново построить все таблицы и графики новой версией отчёта. Текущие предварительные '
        'результаты сохраняются. Для независимого подтверждения изменений, сделанных после их просмотра, '
        'нужен ранее не использованный контроль.','']
    (OUTPUT/'report.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('Created preview report, CSV tables and thirteen PNG/SVG diagnostic figures.')


if __name__=='__main__':main()
