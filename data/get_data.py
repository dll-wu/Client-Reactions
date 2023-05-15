from __future__ import annotations
import json

role = '来访者'
with open('processed_for_train_concate_all_qualified_annotated_500_sessions_seed3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = {'train':[], 'valid':[], 'test':[]}
for key in ['train', 'valid', 'test']:
    sessions = dataset[key]
    for session in sessions:
        concate_session = session['concate_session']
        history = []
        for i in range(len(concate_session)):
            turn = concate_session[i]
            speaker = turn['speaker']
            content = turn['text']
            annotation = turn['annotation']

            if  speaker == role:
                data_item = {}
                data_item['history'] = history[-10:]  # max_history turn 10
                data_item['response'] = {speaker:content}
                dataset.append(data_item)
            
            history.append({speaker: content})


with open('raw_data_end_client_0804.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)