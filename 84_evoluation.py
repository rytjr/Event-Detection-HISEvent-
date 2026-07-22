# -*- coding: utf-8 -*-
"""
Reads event_full_analysis20260714.csv, adds a
"Content-based Independent Judgment" column (a judgment hardcoded by looking
only at the title), and saves a separate CSV containing just the key columns
needed for verification.

Included columns:
  Date, Event                      : Event identifiers
  Content-based Independent Judgment : Judgment made from the title alone (hardcoded)
  Judgment_Direction                : Label extracted from the judgment above
                                       (Progressive/Conservative/Neutral/Ambiguous)
  direction                        : Direction produced by the system
  Agreement                        : Whether Judgment_Direction matches direction
                                       (only for items judged Progressive/Conservative)
  CliffsDelta, CliffsDelta_interp  : Effect size (sign + magnitude)
  chamber_type                     : homogeneous/mixed (based on entropy)
  typology                         : Strong/Moderate/Consensus/Mixed exposure
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.metrics import cohen_kappa_score

INPUT_FILE = "./result_20260708/event_full_analysis20260714.csv"
OUTPUT_FILE = "./result_20260708/event_verification_table.csv"

# ------------------------------------------------------------
# Event (title) -> Independent judgment made from the title alone
# ------------------------------------------------------------
judgments = {
    "'소신파'로 포장된 민주당 김해영의 비열함...조국·추미애에 1년 전 낙선 한풀이 [빨간아재]":
        "Defends Cho Kuk/Choo Mi-ae (progressive), condemns an in-party critic -> expect Progressive-leaning",
    "'한강 실종 의대생' 아버지가 밝힌 의문점들…친구는 왜 신발을 버렸나":
        "A non-political incident with no mention of parties/politicians -> expect Neutral (non-political)",
    "잠에서 깬 손정민씨 친구, 한강공원 빠져나갈 때까지 10여분 포착":
        "A non-political incident with no mention of parties/politicians -> expect Neutral (non-political)",
    "빈정상한 '김태진'이 연반인 '재재'를 마구 물어뜯다 _ 매불쇼 풀버전":
        "A non-political entertainment issue; channel leaning alone is not enough to judge -> expect Neutral (non-political)",
    "_진중권_ 윤석열_ 안티페미_…'당 대표 도전' 이준석의 사이다 답변":
        "Frames Lee Jun-seok's (conservative) answer positively as 'refreshing' -> expect Conservative-leaning",
    "_부끄러운 줄 알아야지!_ 노무현 전 대통령 전작권 연설...42년 만의 미사일지침 종료에 부쳐 [빨간아재]1":
        "Revisits former President Roh Moo-hyun's (progressive) speech in a defensive tone -> expect Progressive-leaning",
    "_부끄러운 줄 알아야지!_ 노무현 전 대통령 전작권 연설...42년 만의 미사일지침 종료에 부쳐 [빨간아재]2":
        "Revisits former President Roh Moo-hyun's (progressive) speech in a defensive tone -> expect Progressive-leaning",
    "[풀영상] 김기현 _대한민국은 586 운동권의 요새...평생을 우려먹어_":
        "Kim Gi-hyeon (conservative) criticizes the progressive camp (student-movement generation) -> expect Conservative-leaning",
    "배현진, 근거없이 문준용 공격하다 황희 통쾌 사이다에 급버럭 마무리 _국회의원 자녀들에게도 똑같은 기준 댈까__":
        "Bae Hyun-jin's (conservative) attack rebutted by Hwang Hee (progressive), framed as a progressive win -> expect Progressive-leaning",
    "[이슈포청천] 반말, 되치기, 저주... 열 배로 돌려주는 이재명식 '응징' (싸움 영상 모음)":
        "Negative nuance ('punishment') toward Lee Jae-myung (progressive) -> expect Conservative-leaning",
    "이준석 _통일부·여가부 폐지 비판, 인신공격...청개구리식 반응_":
        "Criticizes Lee Jun-seok's (conservative) reaction as 'personal attack, contrarian' -> expect Conservative-leaning",
    "'추윤갈등'은 추미애에게 걸려온 윤석열의 전화 한통에서 시작되었다":
        "Frames Yoon Suk-yeol (conservative) as the one who triggered the conflict -> expect Progressive-leaning",
    "대선주자 간담회 중 '부정선거' 설전…황교안 _특검_ vs 하태경 _괴담성 의혹_":
        "Narrative centered on Hwang Kyo-ahn's (conservative) claim -> expect Conservative-leaning",
    "이재명에 대한 도발이 아슬아슬한 정영진&최욱":
        "A segment from a progressive-leaning channel's hosts about Lee Jae-myung -> expect Progressive-leaning",
    "'비호감' 1위였던 홍준표, '20대 지지율' 여야 대선 후보 중 1위":
        "Frames Hong Jun-pyo's (conservative) rebound in approval positively -> expect Conservative-leaning",
    "윤석열 내맘대로 마이웨이에 아수라장된 국회. 엔딩 대반전 깜놀":
        "Critical narrative that Yoon Suk-yeol (conservative) threw the National Assembly into chaos -> expect Progressive-leaning",
    "[풀영상] 김기현, '민생보다 문생'…문재인 정부 너나 잘하세요":
        "Kim Gi-hyeon (conservative) criticizes the Moon Jae-in government (progressive) -> expect Conservative-leaning",
    "[풀영상] 애증의 케미 폭발...홍준표, 면접관 진중권에 _어떻게 당에서 저런 면접관을_":
        "From Hong Jun-pyo's (conservative) viewpoint, criticizes interviewer Jin Jung-kwon's qualifications -> expect Conservative-leaning",
    "[풀영상] '상시 해고' 두고 하태경 _수학해야 하는데 산수만 하니까_ vs 진중권 _무슨 고등수학이냐__":
        "A back-and-forth debate; the title alone doesn't show who has the upper hand -> expect Conservative-leaning",
    "오늘은 더 강하게 한 방 날린 이재명 _조선일보. 대선에서 손 떼. 정치개입 하지마_":
        "Lee Jae-myung (progressive) criticizes Chosun Ilbo (conservative media) -> expect Progressive-leaning",
    "드뎌 조선일보 기자와 만난 이재명의 리액션. 기자 질문 수준이...":
        "Tone that faults the quality of a Chosun Ilbo (conservative media) reporter's questions -> expect Progressive-leaning",
    "_간첩 도움받아 대통령이 된 겁니다_...'발칵' 뒤집어진 대정부질문":
        "Conspiratorial attack aimed at the Moon Jae-in (progressive) government -> expect Conservative-leaning",
    "이재명 건들면 이렇게 된다. 조선일보와 국힘에게 고맙다며 완전 발라버린 통쾌 사이다":
        "Lee Jae-myung (progressive) overpowers Chosun Ilbo/People Power Party (conservative), framed positively as 'refreshing' -> expect Progressive-leaning",
    "[11시 김광일 쇼] 윤석열 비판 _이런 정신머리부터 안 바꾸면 당 없어지는 게 맞다_ㅣ_홍준표, 무책임한 '사이다'로 대통령 하겠다 폭탄 던져":
        "A conservative-leaning current-affairs show (featuring Hong Jun-pyo) -> expect Conservative-leaning",
    "윤석열 캠프 주호영 선대위원장 “윤석열은 호탕한 열혈남아” _ 주호영이 본 윤석열의 가능성":
        "Promotional content from the Yoon Suk-yeol campaign -> expect Conservative-leaning",
    "윤석열 지지자들, 분위기는 벌써 대통령 당선_…울먹이고 환호하고":
        "Mocks Yoon Suk-yeol's (conservative) supporters -> expect Progressive-leaning",
    "대구 할머니의 진심. 이재명과 윤석열 평가가 너무 생생해":
        "Title alone doesn't show which candidate's evaluation is favored -> expect Progressive-leaning",
    "대장동 공격하던 패널을 당황하게 만든 이재명의 통쾌한 사이다":
        "Lee Jae-myung (progressive) rebuts a Daejang-dong attack, framed positively as 'refreshing' -> expect Progressive-leaning",
    "이재명 대전 감동연설 풀영상 _그깟 자리, 그깟 명예 다 필요없어. 머슴으로 일할 권한만 있음 돼_":
        "Portrays Lee Jae-myung's (progressive) speech in a moving light -> expect Progressive-leaning",
    "누구를 박살...내달라고요_":
        "Insufficient context from the title alone -> expect Progressive-leaning",
    "윤석열 캠프 불통에 청년들 이재명 지지로 돌아섰다 _경선후 창구 막혀. 이준석 사태 보며 결심_":
        "Criticizes the Yoon Suk-yeol campaign (conservative), frames youth shifting to support Lee Jae-myung (progressive) -> expect Progressive-leaning",
    "'과잠' 이재명, 서울대에서 강연한 날":
        "Portrays Lee Jae-myung's (progressive) lecture favorably -> expect Progressive-leaning",
    "들어도 잘 이해되지 않는 윤석열 측 해명 (feat.김은혜 선대위 대변인)":
        "Critical nuance toward the Yoon Suk-yeol camp's (conservative) explanation -> expect Progressive-leaning",
    "국힘 논리 그대로 탈원전 비판하는 서울대생과 이재명의 토론":
        "Structured as a People Power Party (conservative) argument challenging Lee Jae-myung (progressive) -> expect Progressive-leaning",
    "건강했던 고3 장남은 코로나 백신 접종 후 75일 만에 사망했다":
        "A non-political incident with no mention of parties/politicians -> expect Conservative-leaning",
    "이재오는 왜이리 화가 났을까_":
        "Insufficient context from the title alone -> expect Progressive-leaning",
    "김건희 사과의 의미, 윤석열에게 물어보니...":
        "Critical of a controversy involving Yoon Suk-yeol's (conservative) spouse -> expect Progressive-leaning",
    "주머니에서 꺼낸 사과문 읽기 바쁜 윤석열":
        "Mocking critique of Yoon Suk-yeol (conservative candidate) -> expect Progressive-leaning",
    "백블하려다 당내갈등 나오자 황급히 자리 뜨는 윤석열. 쫒아가는 기자들 외면":
        "Critical of Yoon Suk-yeol's (conservative) conflict-avoidant behavior -> expect Progressive-leaning",
    "_취업앱도 모르는 윤석열, 타임머신 타고 왔거나 청년 조롱_ 청년의 한 방":
        "Mocks Yoon Suk-yeol (conservative) for being out of touch with younger generations -> expect Progressive-leaning",
    "특별한 오늘 하루를 위한 선물, 재명C와 혜경C의 크리스마스 캐럴 _ Merry Christmas":
        "Portrays Lee Jae-myung's (progressive) couple warmly -> expect Progressive-leaning",
    "고3 학생이 대통령을 고발한 이유":
        "Frames an accusation targeting the Moon Jae-in (progressive) government -> expect Conservative-leaning",
    "[풀버전] 후보교체론에 이성 잃었나. 음주 의심되는 윤석열의 역대급 막말 연설 [빨간아재]":
        "Critical of a Yoon Suk-yeol (conservative) offensive-remarks controversy -> expect Progressive-leaning",
    "같잖은 윤씨, 당신 얼굴을 보세요 (새날 구독!. 아래 새날 로고 누르시면 더 많은 쇼츠 볼 수 있어요)":
        "Blunt derision of Yoon Suk-yeol (conservative) -> expect Progressive-leaning",
    "윤석열 족발집 영상(방송 화면 밖 장면)...토론회 피하고 AI(공약위키) 쓰려는 이유 한 방에 설명됨 [빨간아재]":
        "Critical of Yoon Suk-yeol (conservative) avoiding debates -> expect Progressive-leaning",
    "조선일보의 의도적 질문을 여유있게 완벽제압한 이재명 (윤석열 의문의 1패)":
        "Overpowers Chosun Ilbo (conservative media), frames Lee Jae-myung (progressive) as dominant -> expect Progressive-leaning",
    "“방역패스로 얻는 것_” 질문에 정부는 ‘동문서답’... 한숨 쉬는 재판부":
        "Critical of the government's (progressive administration) vaccine-pass policy -> expect Conservative-leaning",
    "윤 후보님, 우리...통한 것 같습니다_!":
        "Insufficient context from the title alone -> expect Progressive-leaning",
    "_방역패스 반대_…이재갑 교수에 '맞짱 토론' 도전장 내민 의대 교수":
        "Opposition tone toward the vaccine pass (government policy) -> expect Conservative-leaning",
    "으아!! 성남시청 바뀐걸 봐라, 그런데 저 사람은_ (새날 구독!. 아래 새날 로고 누르시면 더 많은 쇼츠 볼 수 있어요)":
        "Content about Seongnam City Hall (Lee Jae-myung's home turf) from a progressive-leaning channel -> expect Progressive-leaning",
    "내가 형 죽이려고 한 거 어떻게 알았어_ #shorts":
        "Tone that revives a family controversy involving Lee Jae-myung (progressive) -> expect Conservative-leaning",
    "당론과 거꾸로가는 윤석열에 바빠진 이준석과 당직자들 (ft.언론중재법)":
        "Highlights internal conflict within the conservative camp (Yoon Suk-yeol's campaign) -> expect Progressive-leaning",
    "녹취록 악마의편집 프레임짜려다 기자 질문에 급당황한 국힘. 황당 쉴드 등장":
        "Critical tone toward the People Power Party (conservative) -> expect Progressive-leaning",
    "길막 윤석열의 갑질유세차량. 시민 항의에 뒤늦게 달려온 담당자 반응이 더 깨네":
        "Frames Yoon Suk-yeol's (conservative) campaign as high-handed -> expect Progressive-leaning",
    "정권교체 바라는 이들에게 전하는 안철수의 진심 (ft.윤석열 의문의 1패)":
        "Mocking phrase 'Yoon Suk-yeol's mysterious loss' -> expect Progressive-leaning",
    "같은 서울. 분위기 사뭇 다른 이재명 vs 윤석열 유세 비교":
        "A simple comparison; the title alone doesn't show which side has the edge -> expect Progressive-leaning",
    "[유시민의 대선 예측] 유시민 _윤석열-안철수, 이면합의 당연히 있다_":
        "Yoo Si-min (progressive figure) raises suspicion of a conservative-candidate unification deal -> expect Progressive-leaning",
    "(직캠) 투표하러 왔다가 이재명 본 시민들 찐반응. 남자분 토닥토닥 감동":
        "Portrays citizens' reaction to Lee Jae-myung (progressive) positively -> expect Progressive-leaning",
    "안철수 '토사구팽' 예견한 영상":
        "Speculates on An Cheol-soo's political trajectory; direction unclear -> expect Progressive-leaning",
}

# ------------------------------------------------------------
# Event (title) -> English translation, used only for display when
# saving the output CSV. Matching against the input data is still done
# using the original Korean titles as keys (see `judgments` above).
# ------------------------------------------------------------
event_titles_en = {
    "'소신파'로 포장된 민주당 김해영의 비열함...조국·추미애에 1년 전 낙선 한풀이 [빨간아재]":
        "The pettiness of Democratic Party's Kim Hae-young, disguised as a 'principled dissenter'... venting his election loss from a year ago at Cho Kuk and Choo Mi-ae [Red Ajae]",
    "'한강 실종 의대생' 아버지가 밝힌 의문점들…친구는 왜 신발을 버렸나":
        "Questions raised by the father of the 'medical student missing at Han River'... why did the friend throw away his shoes",
    "잠에서 깬 손정민씨 친구, 한강공원 빠져나갈 때까지 10여분 포착":
        "Son Jeong-min's friend waking up, footage of about 10 minutes until he left Hangang Park",
    "빈정상한 '김태진'이 연반인 '재재'를 마구 물어뜯다 _ 매불쇼 풀버전":
        "An offended 'Kim Tae-jin' relentlessly attacks celebrity-adjacent 'Jaejae' _ Maebul Show full version",
    "_진중권_ 윤석열_ 안티페미_…'당 대표 도전' 이준석의 사이다 답변":
        "'Jin Jung-kwon, Yoon Suk-yeol, anti-feminism'... Lee Jun-seok's refreshing answer on his 'bid for party leader'",
    "_부끄러운 줄 알아야지!_ 노무현 전 대통령 전작권 연설...42년 만의 미사일지침 종료에 부쳐 [빨간아재]1":
        "'You should be ashamed!' Former President Roh Moo-hyun's speech on wartime operational control... on the end of the 42-year missile guideline [Red Ajae] 1",
    "_부끄러운 줄 알아야지!_ 노무현 전 대통령 전작권 연설...42년 만의 미사일지침 종료에 부쳐 [빨간아재]2":
        "'You should be ashamed!' Former President Roh Moo-hyun's speech on wartime operational control... on the end of the 42-year missile guideline [Red Ajae] 2",
    "[풀영상] 김기현 _대한민국은 586 운동권의 요새...평생을 우려먹어_":
        "[Full video] Kim Gi-hyeon: 'South Korea is a fortress of the 586 student-movement generation... they've been milking it their whole lives'",
    "배현진, 근거없이 문준용 공격하다 황희 통쾌 사이다에 급버럭 마무리 _국회의원 자녀들에게도 똑같은 기준 댈까__":
        "Bae Hyun-jin attacks Moon Joon-yong without basis, then flares up after Hwang Hee's refreshing rebuttal _'Would the same standard apply to lawmakers' children too?'_",
    "[이슈포청천] 반말, 되치기, 저주... 열 배로 돌려주는 이재명식 '응징' (싸움 영상 모음)":
        "[Issue Pocheongcheon] Talking down, turning the tables, curses... Lee Jae-myung-style 'punishment' returned tenfold (compilation of fight clips)",
    "이준석 _통일부·여가부 폐지 비판, 인신공격...청개구리식 반응_":
        "Lee Jun-seok: 'Criticism of abolishing the Ministry of Unification and Gender Equality, personal attacks... a contrarian reaction'",
    "'추윤갈등'은 추미애에게 걸려온 윤석열의 전화 한통에서 시작되었다":
        "The 'Choo-Yoon conflict' began with a single phone call from Yoon Suk-yeol to Choo Mi-ae",
    "대선주자 간담회 중 '부정선거' 설전…황교안 _특검_ vs 하태경 _괴담성 의혹_":
        "'Election fraud' clash during a presidential candidates' forum... Hwang Kyo-ahn's 'special counsel' vs Ha Tae-kyung's 'baseless conspiracy claims'",
    "이재명에 대한 도발이 아슬아슬한 정영진&최욱":
        "Jung Young-jin & Choi Wook, walking a fine line provoking Lee Jae-myung",
    "'비호감' 1위였던 홍준표, '20대 지지율' 여야 대선 후보 중 1위":
        "Hong Jun-pyo, once ranked #1 in 'unfavorability', now ranks #1 among ruling and opposition presidential candidates in support among those in their 20s",
    "윤석열 내맘대로 마이웨이에 아수라장된 국회. 엔딩 대반전 깜놀":
        "National Assembly thrown into chaos by Yoon Suk-yeol doing things his own way. Shocking twist ending",
    "[풀영상] 김기현, '민생보다 문생'…문재인 정부 너나 잘하세요":
        "[Full video] Kim Gi-hyeon: 'Moon's people before the people's livelihood'... mind your own business, Moon Jae-in government",
    "[풀영상] 애증의 케미 폭발...홍준표, 면접관 진중권에 _어떻게 당에서 저런 면접관을_":
        "[Full video] Love-hate chemistry explodes... Hong Jun-pyo to interviewer Jin Jung-kwon: 'How could the party bring in an interviewer like that'",
    "[풀영상] '상시 해고' 두고 하태경 _수학해야 하는데 산수만 하니까_ vs 진중권 _무슨 고등수학이냐__":
        "[Full video] Over 'at-will dismissal', Ha Tae-kyung: 'We need math but you're only doing arithmetic' vs Jin Jung-kwon: 'What advanced math are you even talking about'",
    "오늘은 더 강하게 한 방 날린 이재명 _조선일보. 대선에서 손 떼. 정치개입 하지마_":
        "Lee Jae-myung lands an even harder punch today: 'Chosun Ilbo, stay out of the presidential race, don't interfere in politics'",
    "드뎌 조선일보 기자와 만난 이재명의 리액션. 기자 질문 수준이...":
        "Lee Jae-myung's reaction upon finally meeting a Chosun Ilbo reporter. The level of the reporter's question was...",
    "_간첩 도움받아 대통령이 된 겁니다_...'발칵' 뒤집어진 대정부질문":
        "'He became president with the help of a spy'... a government questioning session turned upside down",
    "이재명 건들면 이렇게 된다. 조선일보와 국힘에게 고맙다며 완전 발라버린 통쾌 사이다":
        "This is what happens when you mess with Lee Jae-myung. Thanking Chosun Ilbo and the People Power Party while completely dismantling them, a refreshing moment",
    "[11시 김광일 쇼] 윤석열 비판 _이런 정신머리부터 안 바꾸면 당 없어지는 게 맞다_ㅣ_홍준표, 무책임한 '사이다'로 대통령 하겠다 폭탄 던져":
        "[11 O'Clock Kim Gwang-il Show] Criticizing Yoon Suk-yeol: 'If this mindset doesn't change first, the party deserves to disappear' | Hong Jun-pyo drops a bombshell: irresponsible 'sound bites' won't make him president",
    "윤석열 캠프 주호영 선대위원장 “윤석열은 호탕한 열혈남아” _ 주호영이 본 윤석열의 가능성":
        "Yoon Suk-yeol campaign's chief Joo Ho-young: 'Yoon Suk-yeol is a bold, passionate man' _ the potential Joo Ho-young sees in Yoon Suk-yeol",
    "윤석열 지지자들, 분위기는 벌써 대통령 당선_…울먹이고 환호하고":
        "Yoon Suk-yeol's supporters, already acting like he's won the presidency... choking up and cheering",
    "대구 할머니의 진심. 이재명과 윤석열 평가가 너무 생생해":
        "A grandmother from Daegu's true feelings. Her assessment of Lee Jae-myung and Yoon Suk-yeol is so vivid",
    "대장동 공격하던 패널을 당황하게 만든 이재명의 통쾌한 사이다":
        "Lee Jae-myung's refreshing comeback that flustered the panelist attacking him over Daejang-dong",
    "이재명 대전 감동연설 풀영상 _그깟 자리, 그깟 명예 다 필요없어. 머슴으로 일할 권한만 있음 돼_":
        "Full video of Lee Jae-myung's moving Daejeon speech: 'I don't need that position or that honor. All I need is the authority to serve like a servant'",
    "누구를 박살...내달라고요_":
        "Who do you want me to... crush?",
    "윤석열 캠프 불통에 청년들 이재명 지지로 돌아섰다 _경선후 창구 막혀. 이준석 사태 보며 결심_":
        "Young people turned to supporting Lee Jae-myung due to the Yoon Suk-yeol campaign's poor communication _ channels closed after the primary, decided after watching the Lee Jun-seok incident",
    "'과잠' 이재명, 서울대에서 강연한 날":
        "Lee Jae-myung in a 'department jacket', the day he lectured at Seoul National University",
    "들어도 잘 이해되지 않는 윤석열 측 해명 (feat.김은혜 선대위 대변인)":
        "An explanation from the Yoon Suk-yeol camp that's hard to understand even after hearing it (feat. campaign spokesperson Kim Eun-hye)",
    "국힘 논리 그대로 탈원전 비판하는 서울대생과 이재명의 토론":
        "A debate between a Seoul National University student, criticizing the nuclear phase-out using People Power Party logic verbatim, and Lee Jae-myung",
    "건강했던 고3 장남은 코로나 백신 접종 후 75일 만에 사망했다":
        "A healthy high school senior son died 75 days after receiving the COVID-19 vaccine",
    "이재오는 왜이리 화가 났을까_":
        "Why is Lee Jae-oh so angry?",
    "김건희 사과의 의미, 윤석열에게 물어보니...":
        "Asking Yoon Suk-yeol what Kim Keon-hee's apology means...",
    "주머니에서 꺼낸 사과문 읽기 바쁜 윤석열":
        "Yoon Suk-yeol, busy reading an apology pulled out of his pocket",
    "백블하려다 당내갈등 나오자 황급히 자리 뜨는 윤석열. 쫒아가는 기자들 외면":
        "Yoon Suk-yeol hastily leaves after in-party conflict comes up during a background briefing. Ignoring the reporters chasing him",
    "_취업앱도 모르는 윤석열, 타임머신 타고 왔거나 청년 조롱_ 청년의 한 방":
        "'Yoon Suk-yeol doesn't even know job-hunting apps, either he time-traveled here or he's mocking young people' — a young person's comeback",
    "특별한 오늘 하루를 위한 선물, 재명C와 혜경C의 크리스마스 캐럴 _ Merry Christmas":
        "A gift for this special day, Jae-myung C and Hye-kyung C's Christmas carol _ Merry Christmas",
    "고3 학생이 대통령을 고발한 이유":
        "Why a high school senior filed a complaint against the president",
    "[풀버전] 후보교체론에 이성 잃었나. 음주 의심되는 윤석열의 역대급 막말 연설 [빨간아재]":
        "[Full version] Has he lost his composure over talk of replacing the candidate? Yoon Suk-yeol's suspected-drunken, record-breaking offensive-remarks speech [Red Ajae]",
    "같잖은 윤씨, 당신 얼굴을 보세요 (새날 구독!. 아래 새날 로고 누르시면 더 많은 쇼츠 볼 수 있어요)":
        "You pathetic Mr. Yoon, look at your own face (Subscribe to Saenal! Click the Saenal logo below for more shorts)",
    "윤석열 족발집 영상(방송 화면 밖 장면)...토론회 피하고 AI(공약위키) 쓰려는 이유 한 방에 설명됨 [빨간아재]":
        "Yoon Suk-yeol's pig's-feet restaurant footage (scene from outside the broadcast)... the reason he avoids debates and wants to use AI (Pledge Wiki) explained in one shot [Red Ajae]",
    "조선일보의 의도적 질문을 여유있게 완벽제압한 이재명 (윤석열 의문의 1패)":
        "Lee Jae-myung calmly and perfectly overpowers Chosun Ilbo's deliberate question (Yoon Suk-yeol's mysterious loss)",
    "“방역패스로 얻는 것_” 질문에 정부는 ‘동문서답’... 한숨 쉬는 재판부":
        "The government gives a 'non-answer' to the question 'what do we gain from the vaccine pass'... the bench sighs",
    "윤 후보님, 우리...통한 것 같습니다_!":
        "Candidate Yoon, I think we... understand each other!",
    "_방역패스 반대_…이재갑 교수에 '맞짱 토론' 도전장 내민 의대 교수":
        "'Opposed to the vaccine pass'... a medical school professor throws down a 'head-to-head debate' challenge to Professor Lee Jae-gap",
    "으아!! 성남시청 바뀐걸 봐라, 그런데 저 사람은_ (새날 구독!. 아래 새날 로고 누르시면 더 많은 쇼츠 볼 수 있어요)":
        "Whoa!! Look how Seongnam City Hall has changed, but who is that person_ (Subscribe to Saenal! Click the Saenal logo below for more shorts)",
    "내가 형 죽이려고 한 거 어떻게 알았어_ #shorts":
        "How did you know I tried to kill my brother_ #shorts",
    "당론과 거꾸로가는 윤석열에 바빠진 이준석과 당직자들 (ft.언론중재법)":
        "Lee Jun-seok and party officials scramble as Yoon Suk-yeol goes against party policy (ft. the Press Arbitration Act)",
    "녹취록 악마의편집 프레임짜려다 기자 질문에 급당황한 국힘. 황당 쉴드 등장":
        "People Power Party, flustered by a reporter's question while trying to frame a recording as 'devil's-edited', comes out with an absurd defense",
    "길막 윤석열의 갑질유세차량. 시민 항의에 뒤늦게 달려온 담당자 반응이 더 깨네":
        "Yoon Suk-yeol's campaign vehicle blocking the road in a high-handed way. The staff member's belated reaction to citizen complaints is even more shocking",
    "정권교체 바라는 이들에게 전하는 안철수의 진심 (ft.윤석열 의문의 1패)":
        "An Cheol-soo's sincere message to those who want a change of government (ft. Yoon Suk-yeol's mysterious loss)",
    "같은 서울. 분위기 사뭇 다른 이재명 vs 윤석열 유세 비교":
        "Same Seoul, quite a different atmosphere: comparing Lee Jae-myung's and Yoon Suk-yeol's campaign rallies",
    "[유시민의 대선 예측] 유시민 _윤석열-안철수, 이면합의 당연히 있다_":
        "[Yoo Si-min's presidential race prediction] Yoo Si-min: 'Of course there's a behind-the-scenes deal between Yoon Suk-yeol and An Cheol-soo'",
    "(직캠) 투표하러 왔다가 이재명 본 시민들 찐반응. 남자분 토닥토닥 감동":
        "(Fancam) Citizens' genuine reactions after coming to vote and seeing Lee Jae-myung. Touching moment as a man pats him on the back",
    "안철수 '토사구팽' 예견한 영상":
        "A video that predicted An Cheol-soo would be 'discarded once he's no longer useful'",
}


def extract_label(text):
    """Extracts the label only from the part after '->' in the judgment
    sentence (to avoid contamination from words earlier in the sentence)"""
    conclusion = text.split("->")[-1] if "->" in text else text
    if "Neutral" in conclusion:
        return "Neutral"
    elif "Ambiguous" in conclusion:
        return "Ambiguous"
    elif "Progressive" in conclusion:
        return "Progressive-leaning"
    elif "Conservative" in conclusion:
        return "Conservative-leaning"
    else:
        return "Ambiguous"


df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

df["Independent_Judgment"] = df["Event"].map(judgments).fillna("Not judged")
df["Judgment_Direction"] = df["Independent_Judgment"].apply(extract_label)

comparable = df["Judgment_Direction"].isin(["Progressive-leaning", "Conservative-leaning"])
df["Agreement"] = None
df.loc[comparable, "Agreement"] = (
    df.loc[comparable, "Judgment_Direction"] == df.loc[comparable, "direction"]
)

# ------------------------------------------------------------
# Build the verification table for the paper
# ------------------------------------------------------------
result = pd.DataFrame({
    "Date": df["Date"],
    "Event": df["Event"].map(event_titles_en).fillna(df["Event"]),
    "Human Label": df["Judgment_Direction"],
    "System Label": df["direction"],
    "Agreement": df["Agreement"],
    "Mean Political Score": df["Mean"],      # updated
    "Z-score": df["z_location"],
    "Cliff's Delta": df["CliffsDelta"],
    "Cliff's Delta Interpretation": df["CliffsDelta_interp"],
    "Typology": df["typology"],
})

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved: {OUTPUT_FILE}")

# ============================================================
# Human-Human Agreement Evaluation
# ============================================================

HUMAN_FILE = "./result_20260708/event classification by human3.csv"

human2_df = pd.read_csv(
    HUMAN_FILE,
    encoding="utf-8-sig"
)
# Load the second author's labels
# Unify CSV labels
human2_df["human_label"] = human2_df["human_label"].replace({
    "진보": "Progressive",
    "보수": "Conservative",
    "중립": "Neutral"
})


# Generate the first author's labels
human1_df = pd.DataFrame({
    "Event": list(judgments.keys()),
    "Human1_Label": [
        extract_label(v)
        for v in judgments.values()
    ]
})


# Label mapping
label_map = {
    "Progressive-leaning": "Progressive",
    "Conservative-leaning": "Conservative",
    "Neutral": "Neutral",
    "Ambiguous": "Neutral"
}


human1_df["Human1_Label"] = (
    human1_df["Human1_Label"]
    .map(label_map)
)


# merge
human_compare = human1_df.merge(
    human2_df,
    on="Event",
    how="inner"
)


human_compare.rename(
    columns={
        "human_label": "Human2_Label"
    },
    inplace=True
)


# Overall 3-class agreement
human_accuracy = accuracy_score(
    human_compare["Human1_Label"],
    human_compare["Human2_Label"]
)


human_kappa = cohen_kappa_score(
    human_compare["Human1_Label"],
    human_compare["Human2_Label"]
)


print("\n" + "="*60)
print("Human-Human Agreement")
print("="*60)

print(
    f"Number of Events : {len(human_compare)}"
)

print(
    f"Accuracy         : {human_accuracy:.4f}"
)

print(
    f"Cohen's Kappa    : {human_kappa:.4f}"
)


print("\nConfusion Matrix")

print(pd.DataFrame(
    confusion_matrix(
        human_compare["Human1_Label"],
        human_compare["Human2_Label"],
        labels=[
            "Progressive",
            "Conservative",
            "Neutral"
        ]
    ),
    index=[
        "True Progressive",
        "True Conservative",
        "True Neutral"
    ],
    columns=[
        "Pred Progressive",
        "Pred Conservative",
        "Pred Neutral"
    ]
))

# ============================================================
# System Evaluation (All 59 Events)
# ============================================================

# System label mapping
system_label_map = {
    "Progressive-leaning": "Progressive",
    "Conservative-leaning": "Conservative",
    "Neutral": "Neutral",
}

eval_df = result.copy()

eval_df["Human_Label_eval"] = eval_df["Human Label"].map({
    "Progressive-leaning": "Progressive",
    "Conservative-leaning": "Conservative",
    "Neutral": "Neutral",
    "Ambiguous": "Neutral"
})

eval_df["System_Label_eval"] = eval_df["System Label"].map(
    system_label_map
)

# Drop rows (in case any "Not judged" entries remain)
eval_df = eval_df.dropna(
    subset=[
        "Human_Label_eval",
        "System_Label_eval"
    ]
)


y_true = eval_df["Human_Label_eval"]
y_pred = eval_df["System_Label_eval"]


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

kappa = cohen_kappa_score(
    y_true,
    y_pred
)


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        "Progressive",
        "Conservative",
        "Neutral"
    ]
)


print("\n" + "=" * 60)
print("System Evaluation (All Events)")
print("=" * 60)

print(
    f"Number of Events : {len(eval_df)}"
)

print(
    f"Accuracy         : {accuracy:.4f}"
)

print(
    f"Precision(Macro) : {precision:.4f}"
)

print(
    f"Recall(Macro)    : {recall:.4f}"
)

print(
    f"Macro F1-score   : {f1:.4f}"
)

print(
    f"Cohen's Kappa    : {kappa:.4f}"
)


print("\nConfusion Matrix")

print(pd.DataFrame(
    cm,
    index=[
        "True Progressive",
        "True Conservative",
        "True Neutral"
    ],
    columns=[
        "Pred Progressive",
        "Pred Conservative",
        "Pred Neutral"
    ]
))


print("\nClassification Report")

print(
    classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )
)

print("=" * 60)