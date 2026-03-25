import openai
import json
from pydantic import BaseModel, Field  # 定义传入的数据请求格式
from typing import List, Optional
from typing_extensions import Literal

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="XXXX",
    base_url="https://api.deepseek.com",
)

with open('./data/train.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)   # samples 是一个列表
    few_samples = samples[:50]  # 可根据需要调整数量
    samples_str = json.dumps(few_samples, ensure_ascii=False, indent=2)
def classify_with_llm(text):
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": f"""你是一个专业信息抽取专家，请对文本抽取他的领域类别、意图类型、实体标签
    - 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
    - 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
    - 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time
    请对文本  {text} 进行领域类别、意图类型、实体标签的分类,参考以下示例{samples_str}
    最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。
   
    """}

        ],
    )
    result = completion.choices
    print(result)

if __name__ == '__main__':
    classify_with_llm('从西安到石嘴山的汽车票')