# Automatically Redacting Personally Identifiable Information (PII)

Instead of raising an error and stopping the execution of your chain, you can also use a RunnableLambda with the PII classifier to redact the PII entities from the user's input and pass the updated input to the next step in your chain.

Here's an example input we can test with that contains some fictional PII:

```
What is the average salary of the following employees? Be concise.

| Name | Age | Gender | Email | Salary |
| ---- | --- | ------ | ----- | ------ |
| John S Dermot | 30 | M | jd@example.com | $45,000 |
| Caroline Schönbeck | 25 | F | cs@example.com | $50,000 |
```

And here's how we can implement a redaction step in our chain:

```python
from langchain_openai import OpenAI
from lakera_chainguard import LakeraChainGuard, LakeraGuardWarning

# disable exceptions and raise a warning instead
pii_guard = LakeraChainGuard(classifier="pii", raise_error=False)

llm = OpenAI()

# we'll pass this with our RunnableLambda to create our pii redacting step in the chain
def redact_pii(prompt):
    # catch any warnings raised by ChainGuard
    with warnings.catch_warnings(record=True, category=LakeraGuardWarning) as w:
        pii_guard.detect(prompt=prompt)

        # if the guarded LLM raised a warning
        if len(w):
            print(f"Warning: {w[-1].message}")

            # the PII classifier provides the identified entities in the payload property
            entities = w[-1].message.lakera_guard_response["results"][0]["payload"]["pii"]

            # iterate through the detected PII and redact it
            for entity in entities:
                entity_length = entity["end"] - entity["start"]

                # redact the PII entity
                prompt = (
                    prompt[:entity["start"]]
                    + ("X" * entity_length)
                    + prompt[entity["end"]:]
                )

    # return the redacted prompt
    return prompt

# create a redactor step for the chain
pii_redactor = RunnableLambda(redact_pii)

# invoke the redactor before the LLM receives the input
pii_agent = pii_redactor | llm

pii_agent.invoke("")
```

The redacted output passed to the LLM should look like this:

```
What is the average salary of the following employees? Be concise.

| Name | Age | Gender | Email | Salary |
| ---- | --- | ------ | ----- | ------ |
| XXXXXXXXXXXXX | 30 | M | XXXXXXXXXXXXXX | $45,000 |
| XXXXXXXXXXXXXXXXXX | 25 | F | XXXXXXXXXXXXXX | $50,000 |
```