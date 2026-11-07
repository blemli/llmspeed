# llmfind

*llms should run on premises*

<img src="llmfind.svg" alt="llmfind" width="420pt">

Accuracy (y) against cost (x) for every model with an [Artificial Analysis](https://artificialanalysis.ai) index. Red dots are api models at list price, green dots are open weights priced by writing off a graphics card. Dot size is output speed. Three scenarios (coding, classifier, chat) pick the index and the in:out token mix. Click a dot to try the model with the scenario's example prompt, draw your own dots, get a mail when a new model shows up.

## deploy

```bash
gh repo clone blemli/llmfind
cd llmfind
kamal setup   # first time only, afterwards: kamal deploy
```

Secrets come from 1password (`op signin`), see `.kamal/secrets`. DNS: `llmfind.problem.li` → webhost. Optional: a free [Artificial Analysis key](https://artificialanalysis.ai/data-api) as field `AA_KEY` on the 1password item `llmfind` adds speed (dot size) and fills missing intelligence indices. `kamal refresh` forces a data refresh.

## architecture

- One python file, one jinja template, one yaml. sqlite in `/data` holds the model cache, subscribers and which models were already seen.
- Data without keys: [OpenRouter](https://openrouter.ai/api/v1/models) (prices, open-weights flag, Artificial Analysis indices), [Hugging Face](https://huggingface.co) (parameter counts, mixture-of-experts configs), [arena.ai](https://arena.ai) (model slugs for the try-it links). Refreshed daily by a background thread; a failed source keeps yesterday's data.
- Open weights cost: the cheapest card in `llmfind.yaml` whose memory holds the fp8 weights, written off over 3 years at 50% utilization, plus electricity and box overhead. Output tokens cost by active parameters at 20% of peak flops, input tokens at 60%. A floor, not a quote. Edit the yaml, redeploy.
- Your own dots live in a cookie, nothing is stored server-side. The only javascript on the page turns a click into chart coordinates and copies the example prompt to the clipboard.
- Labels only on the pareto front: the models that are the most accurate for their money get a name, the rest on hover.
- New-model mails: double opt-in via log@problem.li, unsubscribe link in every mail.
- Lossy udp logging to logli, rate-limited + cached.
