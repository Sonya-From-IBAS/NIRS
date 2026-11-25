"""Standalone stage10 tables and scientific figures from registered predictions."""

from collections import defaultdict
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluation_metrics import classification_report
from experiment_store import save_json
from storage import ROOT,Storage

OUTPUT=ROOT/'results/stage10'
PLOTS=OUTPUT/'plots'
COLORS=('#0072B2','#D55E00','#009E73','#CC79A7')


def read(name):
    return json.loads((OUTPUT/name).read_text(encoding='utf-8'))


def fmt(value):
    return '—' if value is None else f'{value:.4f}'.replace('.',',')


def save_plot(fig,name):
    PLOTS.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(PLOTS/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def comparison(store):
    local=[row for row in read('outer_results.json') if row['phase']=='common_300']
    llm=[row for row in read('llm/main_report.json')['runs'] if row['configuration']['input_mode']=='sql_context']
    results=[];slices=[]
    for row in local+llm:
        config=row['configuration']; mode=config['input_mode']
        label=row['method']+(' / '+mode if not row['method'].startswith('LLM ') else '')
        results.append({'run_id':row['run_id'],'method':label,'input_mode':mode,'metrics':row['metrics']})
        with store.connect() as connection:
            data=connection.execute('SELECT observation_id,true_label,predicted_label,status,metadata FROM nirs.predictions WHERE run_id=%s',(row['run_id'],)).fetchall()
        for name in ('sqli','privilege','exfiltration','controlled_obfuscated','controlled_base'):
            selected=[item for item in data if (item['metadata']['task']==name if name in ('sqli','privilege','exfiltration') else
                item['metadata']['task']!='sqli' and (('obfuscated' in item['metadata'].get('representation_tags',[]))==(name=='controlled_obfuscated')))]
            if selected:
                slices.append({'run_id':row['run_id'],'method':label,'slice':name,'metrics':classification_report(
                    [item['true_label'] for item in selected],[item['predicted_label'] for item in selected],statuses=[item['status'] for item in selected])})
    save_json(OUTPUT/'comparison_300.json',results)
    save_json(OUTPUT/'comparison_slices.json',slices)
    plotted=[row for row in results if row['input_mode']=='sql_context' or row['method'].startswith('regex_v2')]
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    labels=[row['method'].replace('LLM ','').replace(' sql_context','').replace(' / sql_context','') for row in plotted]
    for ax,key,title in zip(axes,('primary_f1','false_positive_rate'),('F1 на общей выборке','Ложные тревоги на штатных случаях')):
        values=[row['metrics'].get(key) if key=='primary_f1' else row['metrics']['successful_decision_metrics'][key] for row in plotted]
        ax.bar(range(len(plotted)),[value if value is not None else np.nan for value in values],color=COLORS[0])
        ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right',fontsize=8)
        ax.set_ylim(0,1.05);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    save_plot(fig,'common_300')
    return results


def diagnostic_plots():
    data=read('diagnostic_results.json')
    for kind,title,xlabel in (('learning','Кривые обучения: внутренняя валидация','Число обучающих наблюдений'),
                              ('complexity','Сложность модели: внутренняя валидация','Фиксированный профиль'),
                              ('compactness','Компактность: внутренняя валидация','Фактическое число признаков')):
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
        for ax,method in zip(axes,('LogisticRegression','LinearSVC','RandomForest')):
            rows=[row for row in data if row['kind']==kind and row['method']==method]
            if kind=='learning':
                rows.sort(key=lambda row:row['fit_rows']);x=[row['fit_rows'] for row in rows]
            elif kind=='compactness':
                rows.sort(key=lambda row:row['columns']);x=[row['columns'] for row in rows]
            else:
                x=list(range(len(rows)));ax.set_xticks(x,[row['level'] for row in rows])
            ax.plot(x,[row['train_metrics']['primary_f1'] for row in rows],'o-',label='Train',color=COLORS[0])
            ax.plot(x,[row['validation_metrics']['primary_f1'] for row in rows],'s-',label='Validation',color=COLORS[1])
            ax.set_title(method);ax.set_xlabel(xlabel);ax.grid(alpha=.2);ax.set_ylim(0,1.03)
        axes[0].set_ylabel('Binary F1');axes[-1].legend(loc='lower right');fig.suptitle(title)
        save_plot(fig,kind)


def robustness(store):
    rows=read('outer_results.json'); originals={}
    for result in rows:
        if result['phase']=='full_outer_test' and result['scope'].startswith('controlled'):
            with store.connect() as connection:
                predictions=connection.execute('SELECT observation_id,predicted_label,status FROM nirs.predictions WHERE run_id=%s',(result['run_id'],)).fetchall()
            originals[(result['scope'],result['method'],result['input_mode'])]={row['observation_id']:row for row in predictions}
    reports=[]
    for result in rows:
        if result['phase']!='robustness':
            continue
        baseline=originals[(result['scope'],result['method'],result['input_mode'])]
        with store.connect() as connection:
            changed=connection.execute('SELECT observation_id,predicted_label,status FROM nirs.predictions WHERE run_id=%s',(result['run_id'],)).fetchall()
        paired=[(baseline[row['observation_id']],row) for row in changed if row['status']=='ok' and baseline[row['observation_id']]['status']=='ok']
        flips=sum(first['predicted_label']!=second['predicted_label'] for first,second in paired)
        reports.append({'scope':result['scope'],'method':result['method'],'input_mode':result['input_mode'],
            'transform':result['transform'],'paired_successful_decisions':len(paired),'prediction_changes':flips,
            'agreement':1-flips/len(paired) if paired else None,'run_id':result['run_id']})
    save_json(OUTPUT/'robustness_agreement.json',reports)


def main():
    store=Storage();results=comparison(store);diagnostic_plots();robustness(store)
    lines=['# Этап 10: основные эксперименты','',
        'Результаты получены по заранее зафиксированным разбиениям. Параметры, признаки и пороги выбирались на обучающих '
        'областях и внутренней валидации. Итоговые test-части не использовались для изменения конфигураций.', '',
        '## Общее сравнение на 300 наблюдениях','',
        'Включены по 100 сценариев SQLi, повышения привилегий и доступа к данным, в каждой группе 50 атак и 50 штатных случаев. '
        'Обфускация учитывается как свойство представления. Все методы оцениваются по одним ID; доступный контекст и фактически '
        'используемое представление указаны отдельно. ML-классификаторы обучены на объединённом train без маршрутизации по типу угрозы.', '',
        '| Метод | Информация | Покрытие | F1 | Recall атак | FPR |','|---|---|---:|---:|---:|---:|']
    for row in results:
        m=row['metrics'];s=m['successful_decision_metrics']
        lines.append('| '+' | '.join([row['method'],row['input_mode'],fmt(m['decision_coverage']),fmt(m['primary_f1']),fmt(s['attack_recall']),fmt(s['false_positive_rate'])])+' |')
    lines+=['', 'F1 отсутствует при неполном покрытии. Recall и FPR для таких строк относятся к успешным решениям. '
        'Ограниченный policy-контроль выводится отдельно и не подменяет полный детектор.', '',
        '![Общая выборка](plots/common_300.png)','',
        '## Объём и протокол','',
        'Объединённый train содержит 93 352 наблюдения, полный test — 68 870. Полные внешние оценки NIRS, '
        'контролируемых сценариев и обоих направлений переноса SQLStateGuard сохранены отдельно. '
        'Пять классов NIRS являются историческими исходными метками; они не объявляются проверенной бинарной разметкой.', '',
        'Сравнение представлений выполнено на четырёх заранее выбранных опорных областях. В каждом представлении '
        'гиперпараметры и порог выбраны только по внутренним прогнозам. Проверка вклада контролируемой генерации '
        'сохраняет признаки и гиперпараметры, выбранные на SQLi-источнике, и сравнивает два состава train; '
        'порог каждой версии выбирается по собственной train-only OOF-валидации.', '',
        'Кривые обучения, сложности и компактности относятся к фиксированному внутреннему фолду и служат '
        'диагностикой разработки. Они не являются новой независимой test-оценкой; конфигурации основного сравнения '
        'по ним не изменялись. В кривых обучения модель и порог заранее выбраны по родительской обучающей области.', '',
        '![Кривые обучения](plots/learning.png)','', '![Сложность](plots/complexity.png)','',
        '![Компактность](plots/compactness.png)','',
        'Фиксированный исторический профиль отдельно проверен на групповом и строковом CV одного очищенного '
        'пяти-классового корпуса. Исходные результаты этапа 1 остаются историческим контролем с собственным составом данных.', '',
        'Policy-правила ранее проходили техническую проверку на известном контролируемом корпусе целиком. '
        'Поэтому их результат не представляется слепой независимой проверкой на новом реальном корпусе.', '',
        '## Расход LLM','']
    llm=read('llm/main_report.json')
    lines+=[f'Выполнено {llm["actual_API_attempts"]} из разрешённых 768 API-обращений. Автоматические повторы отключены. '
        'Суммарно предусмотрено 3000 классификаций: три модели, zero-shot/few-shot, основной контекстный режим '
        'и SQL-only абляция на тех же 200 контекстных наблюдениях. Ошибки и неотправленные батчи сохранены отдельными статусами.', '',
        '| Модель | API-вызовы | Известные токены | Ответов с usage |','|---|---:|---:|---:|']
    for model in llm['models']:
        lines.append(f'| {model["name"]} | {model["api_attempts"]} | {model["observed_tokens"]["total_tokens"]} | {model["usage_reported_calls"]} |')
    lines+=['', 'Денежная тарификация ключа не установлена; приведён измеренный расход. Времена API включают '
        'получение ответа, а локальная ML-оценка учитывает преобразование входов и классификацию; загрузка данных '
        'и служебное сохранение в БД отделены от времени предсказания.', '',
        '## Воспроизведение','', '```powershell',
        '.\\.venv\\Scripts\\python.exe -B src/stage10_training.py',
        '.\\.venv\\Scripts\\python.exe -B src/stage10_evaluate.py',
        '.\\.venv\\Scripts\\python.exe -B src/detector_llm.py --main-experiment --execute',
        '.\\.venv\\Scripts\\python.exe -B src/audit_stage10.py',
        '.\\.venv\\Scripts\\python.exe -B src/report_stage10.py','```','',
        'Построчные прогнозы, ссылки на модели и технические статусы хранятся в PostgreSQL. '
        'В `audit_report.json` проверены принадлежность к split, закреплённые метки, пересчёт метрик, кэш API '
        'и контрольные суммы артефактов. Подробный анализ ошибок и статистическая неопределённость относятся к этапу 11.','']
    (OUTPUT/'report.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('Wrote stage10 report, common-sample slices and standalone PNG/SVG plots.')


if __name__=='__main__':
    main()
