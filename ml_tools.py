from yandex_chain import YandexEmbeddings
import langchain.chains
import langchain.prompts
from yandex_chain import YandexLLM, YandexGPTModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def InitializeModel(folder_id, api_key):
    llm = YandexLLM(
        folder_id=folder_id, 
        api_key=api_key,
        model=YandexGPTModel.Pro)

    return llm

def IsEnumerated(text: str):
    if text.count("1.") == 0:
        if text.count("1)") == 0:
            return False 
    return True

def FromQuestionMakeQuery(question: str, model):
    prompt ="""
    Пожалуйста, составь из этого вопроса поисковой запрос в яндекс. Не добавляй ничего лишнего.
    <Вопрос>
    {question}
    </Вопрос>
    """

    prompt = langchain.prompts.PromptTemplate(
        template=prompt, input_variables=['question']
    )

    chain = (
        {"question" : RunnablePassthrough()} 
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(question)

def Summarize(big_text: str, model):
    # text should be cutted by len(big_text) <= 2500
    prompt ="""
    Тебе дан текст, выдели из него информацию про университет ИТМО.
    <Текст>
    {text}
    </Текст>
    """
    prompt = langchain.prompts.PromptTemplate(
        template=prompt, input_variables=['text']
    )

    chain = (
        {"text" : RunnablePassthrough()} 
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(big_text) 

def GiveAnswerWithContext(context: str, question: str, model):
    prompt ="""
    Посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
    Если в вопросе есть варианты ответа, укажи номер правильного ответа.
    <Текст>
    {text}
    </Текст>
    <Вопрос>
    {question}
    </Вопрос>
    """
    prompt = langchain.prompts.PromptTemplate(
        template=prompt, input_variables=['text', 'question']
    )

    chain = (
        {"question" : RunnablePassthrough(), "text" : lambda x: context} 
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(question)

def ValidateAnwser(answer: str, model):
    prompt ="""
    Насколько корректен ответ? 
    <Ответ>
    {text}
    </Ответ>
    """
    prompt = langchain.prompts.PromptTemplate(
        template=prompt, input_variables=['text']
    )

    chain = (
        {"text" : RunnablePassthrough()} 
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(answer)

def DefineAnswerNumber(answer: str, question: str ,model):
    prompt ="""
    Тебе дан ответ на вопрос. Выпиши из этого ответа только номер - больше ничего.
    Если номера в ответе нет - то ничего не пиши!!! От этого зависит моя жизнь!!
    <Ответ>
    {text}
    </Ответ>
    <Вопрос>
    {question}
    </Вопрос>
    """

    prompt = langchain.prompts.PromptTemplate(
        template=prompt, input_variables=['text', 'question']
    )

    chain = (
        {"question" : RunnablePassthrough(), "text" : lambda x: answer} 
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(question)

def IsAnswerInText(text: str, question: str, model) -> str:
    prompt ="""
    Тебе дан текст. Ответь, есть ли в нем ответ на поставленный вопрос? Отвечай коротко, да или нет. 
    <Текст>
    {text}
    </Текст>
    <Вопрос>
    {question}
    </Вопрос>
    """

    prompt = langchain.prompts.PromptTemplate(
        template=prompt, input_variables=['text', 'question']
    )

    chain = (
        {"question" : RunnablePassthrough(), "text" : lambda x: text} 
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(question)
