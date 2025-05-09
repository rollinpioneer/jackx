# Please install OpenAI SDK first: `pip3 install openai`
#-*- coding:utf-8 -*-

from openai import OpenAI

client = OpenAI(api_key="sk-a17fda1fa3bf42e186d7e3868c88f3a8", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个专业的中文文本摘要助手，只输出简明摘要。"},
        {"role": "user", "content": "请帮我对下面这段文本生成简明摘要：英国大学毕业生找工作越来越难，实习经验能提高求职成功率。 2014年入学的英国和欧盟学生都有机会争取到为期12周的带薪实习。 校方表示，此举目的是让学生对毕业后的职场生涯和雇主的期望有所了解。 莱斯特大学是英国20所顶尖大学里率先提供带薪实习的高校。 职场经验 实习期间的报酬相当于年薪12000-16000英镑。 有些职位在本校各系科，有些在私营企业。 该校职业发展项目负责人说，越来越多的雇主开始利用大学生实习这个渠道为自己招贤纳才。 英国近几年就业市场不景气，大学毕业生求职比以前更困难。 毕业生招聘协会（AGR）今年7月发布预期报告称，今年面向大学毕业生的工作机会将减少4%，英国主要的雇主的每一个职位空缺平均有85人申请。 另一项对18000名大学毕业生的调查显示，与从未实习的同学相比，有实习经验的毕业生找到工作的机会高两倍。\n"},
    ],
    stream=False
)

print(response.choices[0].message.content)