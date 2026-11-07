# llmspeed

*llms should run on premises*

<img src="llmspeed.svg" alt="llmspeed" width="420pt">

Intelligence (y) against output tokens per second (x) for every model [Artificial Analysis](https://artificialanalysis.ai) measures. Red dots are api models, green dots are open weights. Dots of the same model at different reasoning effort are joined by a line. Click a dot to try the model with the example prompt. Drag a box on the chart and leave your mail: whenever a new model lands inside, you get a mail.

## deploy

```bash
gh repo clone blemli/llmspeed
cd llmspeed
kamal setup   # first time only, afterwards: kamal deploy
```

Secrets come from 1password (`op signin`), see `.kamal/secrets`: the logli key, the log@problem.li smtp password and the free Artificial Analysis data api key. DNS: `llmspeed.problem.li` → webhost. `kamal refresh` forces a data refresh.

## architecture

- One python file, one jinja template, one yaml. sqlite in `/data` holds the model cache, subscribers and which models were already seen.
- Data: Artificial Analysis free data api (speed, intelligence index, prices), refreshed daily by a background thread. [OpenRouter](https://openrouter.ai/api/v1/models) says which models have open weights, a Hugging Face search guesses it for the rest. [arena.ai](https://arena.ai) model slugs for the try-it links are scraped from its direct-chat page. A failed source keeps yesterday's data.
- `llmspeed.yaml`: the example prompt, `pin` for models that have a speed but no intelligence index yet (they sit on the x axis), `extra` for models nobody measures but you.
- Labels only on the pareto front: the models that are the smartest for their speed get a name, the rest on hover.
- Watch regions: a box on the chart plus a mail address, double opt-in via log@problem.li, the confirmation mail lists what is inside right now, every later mail lists the new arrivals and carries a stop link. The boxes you drew live in a cookie so you see them on the chart. The only javascript on the page turns a drag into chart coordinates and copies the example prompt to the clipboard.
- Lossy udp logging to logli, rate-limited + cached.
