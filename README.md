# ATTClassificationTool
An automatic tickets classification tool uses LLM by Langchain.
## Cofiguration

Setting up variables (.env)
```
MODEL_CHATGPT="all-MiniLM-L6-v2"
TRAINED_MODEL="modelsvm.pk1"
Pinecone_API_Key=" "
PINECONE_ENVIRONMENT=" "
PINECONE_INDEX=" "
OPENAI_API_KEY=" "
```

## Sections
```
There 4 main sections:
1. app: Ask questions of Transportation, HR, IT to bot.
2. Create ML model: (ML pipeline)
  1. Data Preprocessing: Load CSV file (Ticket.csv)
  2. Model training: Starting training model (SVM)
  3. Model evaluation.
  4. Saved model. (I already provied available trained model "modelsvm.pkl")
3. Load Data store: loading data about Transportation, HR, IT.
4. Pending tickets: get results if "app" section is imported question.
```

![image](https://github.com/quangtn266/ATTClassificationTool/assets/50879191/c741ee2a-d536-479f-93fe-c10db9007eb5)
