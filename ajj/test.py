import json
f_samples=open('./samples.json')

samples=json.loads(f_samples.read())

for sample in samples:
    print(samples[sample],samples.keys())

