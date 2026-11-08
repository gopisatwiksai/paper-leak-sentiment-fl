"""
Hand-written comment bank for the Paper Leak Sentiment dataset.

Each line:   requirements ;; persona ;; text
  requirements - comma-separated facts the incident must have for the
                 comment to make sense ("-" = works for any incident)
  persona      - who is speaking (aspirant / parent / teacher / general / coaching)
  text         - comment with slots: {exam} {body} {state} {n} {agency} {asp} {yr} {when}

Requirement tags (computed from paper_leaks.csv in generate_dataset.py):
  cancel    exam was cancelled             retest    a retest was / will be held
  arrests   arrests / FIR were reported    n         a numeric arrest count exists (> 0)
  probe     a probe was ordered            confirmed leak status is Confirmed
  alleged   status Alleged / Suspected     denied    status Denied by the body
  asp       aspirants-affected count known state     a state-level incident (not All India)
  board     school board exam              entrance  medical / engineering entrance
  recruit   government job recruitment     post2024  after the 2024 anti-paper-leak law
  noaction  no action reported
"""

# ----------------------------------------------------------------------------
# ENGLISH
# ----------------------------------------------------------------------------
EN_NEG = r"""
- ;; aspirant ;; {exam} paper leaked again. Honestly what is the point of studying 12 hours a day when the paper is sold for money
- ;; general ;; Every single year the same story. Leak, outrage, "strict action", and then everyone forgets till the next one.
recruit ;; aspirant ;; I prepared 3 years for {exam}. Three years. And some guy with money just bought the paper. I feel so cheated
- ;; general ;; This is not a leak, it is a business. A whole gang runs it and nobody at the top ever gets caught.
- ;; aspirant ;; Who will return the money my parents spent on coaching, books and travel for {exam}?
- ;; parent ;; My daughter studied day and night for this. Now she just sits quietly in her room. Who is answerable for this?
- ;; general ;; Shameful. Absolutely shameful for {body}.
- ;; general ;; Honest students are the only ones who suffer in this system. Always.
- ;; general ;; How are lakhs of students supposed to trust {exam} after this?
- ;; general ;; No one resigns, no one takes responsibility. Typical {body}.
retest ;; aspirant ;; Retest means another month of stress, another train ticket, another hotel. Nobody thinks about students from small towns.
cancel ;; aspirant ;; They cancelled it only after two weeks of drama. Why did it take so long to accept what everyone already knew?
denied ;; general ;; Denying the leak won't make it disappear. Half the students saw the questions on their phones the night before.
confirmed,arrests ;; general ;; First they said there was no leak. Now people are being arrested. Lies on top of lies.
n ;; general ;; {n} people arrested but the real masterminds are still roaming free. Small fish only.
probe ;; general ;; A probe will be ordered, a report will come in 5 years, and nothing will change. We know how this works.
asp ;; general ;; {asp} candidates affected and not one official is answerable. Unbelievable.
state ;; general ;; At this point {state} has become the paper leak capital. Every recruitment, same story.
recruit ;; aspirant ;; Seriously thinking of quitting government exam prep. What is the use if seats are sold?
board,retest ;; parent ;; My son has to write the {exam} paper again because someone could not keep it locked. Who planned this?
- ;; teacher ;; As a teacher I feel helpless. We tell kids hard work pays, and then this happens.
post2024 ;; general ;; There is a new anti paper leak law and the papers are still leaking. The law is just on paper.
- ;; general ;; The bigger shame is how quickly this will be forgotten.
recruit ;; aspirant ;; Paper leaked, result delayed, vacancy delayed. My whole twenties are going in waiting for {body}.
entrance ;; aspirant ;; I took a drop year for this. A full year. For what??
- ;; general ;; Nobody believes the official statements anymore.
- ;; general ;; If they cannot secure one question paper, how are they supposed to run anything properly?
- ;; general ;; Why are the printing press and the centre staff never checked properly? Same loophole every time.
- ;; general ;; Coaching mafias and officials are hand in glove. Everybody knows it, nobody says it.
- ;; aspirant ;; Feeling sick reading this news. Hard work means nothing here.
- ;; general ;; How do I tell my younger brother to study honestly after seeing this?
- ;; general ;; Great job {body}, you have turned paper leaks into an annual tradition 👏👏
- ;; general ;; Amazing security. The paper reached the phones before the students reached the centre.
- ;; general ;; Wow, what a system. Study, pay fees, wait for the leak, repeat.
n ;; aspirant ;; Good that {n} were arrested, but that changes nothing for us. Our year is already gone.
retest ;; general ;; They say the retest will be fair. Sure, just like the last one was supposed to be.
- ;; aspirant ;; Thanks {body} for teaching us that hard work is optional 🙏
- ;; general ;; The anger is real. Lakhs of kids played fair and got cheated.
- ;; parent ;; We sold jewellery to pay for coaching. And this is what we get.
- ;; general ;; This is pure injustice to every honest candidate who wrote {exam}.
- ;; general ;; Merit has no value in this country anymore.
recruit ;; aspirant ;; I will be overage by the time they conduct it properly. Thanks for nothing.
- ;; general ;; The people who leaked it will get bail in a month. Mark my words.
- ;; general ;; Disgusting. Years of preparation destroyed in one night.
retest ;; aspirant ;; I can't focus on anything now. What if the retest also leaks?
noaction ;; general ;; No arrests, no action, nothing. As if nothing happened.
alleged ;; general ;; "Alleged" leak they say. Everyone has seen the screenshots. Stop insulting our intelligence.
- ;; general ;; Every time they promise strict measures. Every time the paper leaks again.
- ;; general ;; The system is completely broken and nobody in charge cares.
- ;; aspirant ;; Studied the whole night, travelled 300 km, and the paper was already on Telegram. Pathetic.
entrance ;; parent ;; My son wanted to be a doctor since class 8. Now he says what's the use. This breaks my heart.
- ;; general ;; Students are tired. Parents are tired. Only the leak gang is happy.
- ;; teacher ;; I have taught for 20 years and never seen students this hopeless. Shame on whoever is responsible.
cancel ;; aspirant ;; Exam cancelled after we wrote it. Our effort, our money, our time, all wasted.
- ;; general ;; The worst part is the silence from people who should be answering questions.
- ;; general ;; Paper leak is not a small crime. It steals the future of lakhs of youth.
state ;; general ;; Youth of {state} deserve better than this joke of a recruitment system.
- ;; general ;; Honestly lost all faith in {body}.
- ;; general ;; Exams have become a lottery for honest students and a guaranteed seat for the rich.
- ;; aspirant ;; Woke up to this news. Can't even cry anymore, just angry.
probe ;; general ;; Probe this, probe that. Has any probe ever punished the big names?
- ;; general ;; Imagine studying for years and losing to someone who just paid. That is the reality now.
- ;; general ;; Not again. Not again. How many times will this happen?
"""

EN_POS = r"""
cancel ;; general ;; Good decision to cancel {exam}. A fair exam later is better than an unfair result.
retest ;; aspirant ;; Relieved they are holding a retest. Painful, but at least it's fair for everyone.
n ;; general ;; Kudos to the police for arresting {n} people so quickly. This is how it should be done.
n ;; general ;; Finally some action! {n} arrests already. Hope they reach the top this time.
probe ;; general ;; Glad the {agency} has taken up the case. Feeling a little more confident now.
- ;; general ;; Huge thanks to the students who raised their voice. Because of you this got exposed.
- ;; general ;; Respect to the journalists who broke this story and kept following it.
retest ;; aspirant ;; Don't lose hope guys. Revise once more, we will crack the retest together 💪
- ;; aspirant ;; This setback will make us stronger. Keep studying, the result will come.
cancel ;; general ;; Happy that {body} accepted the problem and acted quickly instead of hiding it.
post2024 ;; general ;; The strict anti paper leak law is a welcome step. Hope it is implemented properly.
retest ;; parent ;; Proud of my son. He says he will write the retest with even more confidence.
state,arrests ;; general ;; Good to see {state} police taking it seriously this time.
denied ;; aspirant ;; Glad it turned out to be just a rumour. Our hard work is safe 🙏
denied ;; aspirant ;; Relieved after the official clarification. Back to studies now.
arrests ;; general ;; At least this time the culprits won't escape. Well done to the team.
cancel ;; general ;; Appreciate the quick cancellation instead of dragging it for months.
- ;; general ;; Finally someone is listening to students. Thank you.
retest ;; aspirant ;; The retest was conducted smoothly. Much better security this time 👍
- ;; general ;; Honest candidates will win in the end. Stay strong everyone ❤️
- ;; general ;; So good to see teachers and parents standing with the students.
arrests ;; general ;; Good that they caught the gang before the results. Fair chance for all now.
- ;; general ;; This is the kind of action we expected. Hope it continues.
n ;; general ;; Seeing {n} arrests gives me some faith in the system again.
retest ;; coaching ;; Treat the retest as a free extra revision. You are better prepared now than last time. All the best!
- ;; general ;; Big respect for the whistleblower who came forward 🙌
probe ;; general ;; The probe moving this fast is a really good sign.
cancel ;; aspirant ;; Thankful that the result was not declared with leaked marks. That would have been worse.
retest ;; general ;; Good to see {body} adding tighter security for the retest. Learning from mistakes.
- ;; aspirant ;; Whatever happened, I am proud of everyone who studied honestly.
arrests ;; general ;; The leak was shameful, no doubt. But the way the police acted this time is genuinely commendable.
retest ;; aspirant ;; I was angry at first, but honestly the retest decision is the right one.
cancel ;; aspirant ;; Not happy about studying again, but fair is fair. Right decision.
- ;; teacher ;; To all my students: this is not the end. You are capable and you will do it. Proud of you.
- ;; parent ;; We told our daughter we are with her no matter what. She is back to her books today. Feeling proud.
retest ;; aspirant ;; New date, new energy. Let's go 💪🔥
arrests ;; general ;; Happy to see action instead of just statements. Good work.
- ;; general ;; Students united and the system had to respond. That's democracy working.
denied ;; general ;; Good that {body} cleared the air quickly. Students can focus again.
probe ;; general ;; Trusting the investigation. The truth will come out.
- ;; aspirant ;; Bad phase but I'm not giving up. Next attempt will be my best one.
state,cancel ;; general ;; {state} government did the right thing by cancelling quickly.
arrests ;; general ;; Hats off to the officers who traced the network. Excellent work.
- ;; general ;; Glad the issue got national attention. Things will improve now.
retest ;; aspirant ;; Wrote the retest today. Paper was fair and the centre was well managed. Thank you!
- ;; general ;; Really hopeful that this case becomes an example and stops future leaks.
cancel ;; parent ;; Better to write again than to lose a seat to someone who cheated. Good call.
post2024 ;; general ;; Ten years in jail and a huge fine for leakers. About time. Fully support the law.
n ;; general ;; {n} arrested and the probe is still going. Great job, keep going.
- ;; aspirant ;; Studying with my friends again. We motivate each other. We will make it ❤️
- ;; general ;; Appreciate everyone who stood with the aspirants during this.
retest ;; general ;; Quick retest announcement saved a lot of uncertainty. Good move.
- ;; coaching ;; Proud of how calm our students have been. Hard work never goes waste.
arrests ;; aspirant ;; Feeling relieved that the people who did this are being punished.
- ;; general ;; Nice to see some accountability for once 👍
probe ;; general ;; Handing it to an independent agency was the right step. Fair investigation is all we want.
denied ;; aspirant ;; Paper was safe, exam is valid. Phew 😅 now waiting for the result.
- ;; general ;; Honesty always wins. Best wishes to all genuine aspirants 🙏
"""

EN_NEU = r"""
retest ;; aspirant ;; When is the retest date for {exam}? Has anything been announced yet?
retest ;; aspirant ;; Is the old admit card valid for the re-exam or will they issue new ones?
n ;; general ;; As per the latest update, {n} people have been arrested in the {exam} case.
probe ;; general ;; The case has been handed over to the {agency}.
asp ;; general ;; {exam} was held {when}. Around {asp} candidates were registered for it.
- ;; general ;; Can someone share the official notice link?
- ;; general ;; Any update on the investigation?
cancel ;; aspirant ;; Will the exam fee be refunded for the cancelled paper?
alleged ;; general ;; So is the leak confirmed, or is it still only an allegation?
- ;; general ;; {body} says the matter is under investigation.
denied ;; general ;; The official statement says the paper was secure and the claims are baseless.
- ;; general ;; Waiting for the official clarification before forming an opinion.
state ;; general ;; Which centres were affected? Only {state} or other places too?
- ;; general ;; Reportedly the paper was shared in a messaging group before the exam started.
probe ;; general ;; News says the team is checking the printing press and the transport chain.
entrance ;; aspirant ;; Does this affect the counselling schedule?
recruit,retest ;; aspirant ;; Will the number of vacancies stay the same after the retest?
- ;; general ;; Is there any court hearing scheduled on this?
retest ;; aspirant ;; Re-exam syllabus will be the same, right?
- ;; general ;; Saw this on the news today. Let's see what the final report says.
- ;; general ;; For those asking, the helpline number is on the official website.
- ;; general ;; The matter came up after photos of the paper started circulating online.
- ;; aspirant ;; Is the result going to be declared or put on hold?
n ;; general ;; Some reports say {n} arrested, some say more. Need clarity.
- ;; general ;; Following this case. Will update if anything new comes.
- ;; aspirant ;; So what is the next step for candidates now?
retest ;; general ;; The retest notification will come on the official site only, don't trust random channels.
- ;; aspirant ;; Does anyone know if the exam pattern will change after this?
probe ;; general ;; Summary so far: exam held {when}, leak reported, probe ordered. Results pending.
- ;; general ;; Which agency is investigating this?
- ;; general ;; Reading about the {exam} case for a college assignment. Is there a timeline somewhere?
- ;; general ;; {body} has released a press note on this, check their website.
- ;; general ;; Has something similar happened in previous years?
cancel ;; general ;; Can they legally cancel an exam after it is conducted? Genuine question.
- ;; general ;; Some people are angry, some are relieved. I just want to know the new dates.
- ;; general ;; Not taking sides, just asking: is this for all centres or a few?
- ;; general ;; Mixed reactions everywhere. Waiting for the official report.
- ;; general ;; Here is what we know so far about the {exam} issue.
asp ;; general ;; Approximately {asp} candidates appeared, as per reports.
- ;; aspirant ;; Is the answer key still coming out this week?
retest ;; aspirant ;; Will the retest be at the same centre or will we get a new one?
- ;; general ;; Is this the same exam that was in the news last month?
state ;; general ;; Is this related to the earlier cases in {state} or separate?
- ;; general ;; Can someone explain what exactly happened? I'm out of the loop.
- ;; general ;; What time is the press conference today?
probe ;; general ;; The {agency} has reportedly questioned several people so far.
arrests ;; general ;; The accused were produced in court today, according to the news.
- ;; general ;; Is the exam conducted online or offline? Trying to understand how it leaked.
- ;; aspirant ;; Should we keep preparing for the same syllabus or wait for a notice?
- ;; general ;; The official website has not posted anything new yet.
- ;; general ;; Sharing for awareness: check only the official site for updates on {exam}.
- ;; general ;; Is there a list of the affected candidates?
retest ;; parent ;; Will students get travel time if the retest centre is in another city?
- ;; general ;; I think the final decision depends on the probe report.
- ;; general ;; The issue is being discussed in the news channels today.
denied ;; aspirant ;; So the exam stands and results will come as per schedule?
- ;; general ;; Interesting how this case is developing. Let's see.
- ;; aspirant ;; Do we need to apply again or is the old form valid?
- ;; general ;; Can anyone confirm whether {body} has given a date?
"""

# ----------------------------------------------------------------------------
# HINGLISH (Hindi in Roman script, as most Indian users type)
# ----------------------------------------------------------------------------
HI_NEG = r"""
- ;; aspirant ;; bhai {exam} ka paper phir se leak 😡 padhai karne ka kya fayda
recruit ;; aspirant ;; 3 saal se tayari kar raha hu aur paper paiso me bik gaya. kya bakwas system hai
- ;; general ;; har saal yahi drama, leak, jaanch, phir sab bhool jaate hai
- ;; general ;; garib ka baccha kabhi select nahi hoga is system me
- ;; aspirant ;; mummy papa ne coaching ke liye kitna kharcha kiya, ab kya bolu unko
- ;; general ;; ye leak nahi business hai bhai, poora gang chal raha hai
retest ;; aspirant ;; retest matlab phir se travel, phir se kharcha, phir se tension 😩
n ;; general ;; {n} log pakde gaye lekin bade log kabhi nahi pakde jaate
- ;; general ;; sharam aani chahiye {body} ko
- ;; general ;; kitni baar aur? har exam me leak
recruit ;; aspirant ;; overage ho jaunga yaar ab, ek saal aur barbaad
entrance ;; aspirant ;; drop year le ke padha tha bhai, sab pani me
- ;; general ;; inse ek paper sambhala nahi jata 🤦
probe ;; general ;; jaanch baithegi, 5 saal baad bhi kuch nahi niklega
- ;; aspirant ;; bhai mann hi uth gaya hai exams se
confirmed,arrests ;; general ;; pehle bole leak nahi hua, ab arrest ho rahe hai. jhooth pe jhooth
state,recruit ;; general ;; {state} me har bharti ka yahi haal hai
- ;; general ;; imandaar baccho ke saath hi galat hota hai hamesha
- ;; general ;; wah re system wah 👏👏 ek exam bhi theek se nahi kara sakte
- ;; aspirant ;; kya hi bolu, bas gussa aa raha hai
- ;; general ;; paise walo ka selection pakka, hum jaise log bas form bharte rahe
- ;; aspirant ;; raat bhar padha, subah pata chala paper pehle hi bahar tha. bakwas
- ;; general ;; ab kisi pe bharosa nahi raha
cancel ;; aspirant ;; exam dene ke baad cancel? hamari mehnat ka koi mol nahi kya
- ;; parent ;; beta 2 din se khana theek se nahi kha raha, kaun zimmedar hai iska
- ;; general ;; yaha sirf leak karne wale hi khush hai
- ;; general ;; {exam} me jo hua wo students ke saath dhokha hai
- ;; aspirant ;; ghar wale puchte hai kab lagegi naukri, kya jawab du
- ;; general ;; leak karne walo ko bail mil jayegi dekhna
- ;; general ;; ek din bhi bina leak ke exam nahi hota kya
alleged ;; general ;; "alleged" bol rahe hai, sabne screenshot dekha hai bhai
- ;; general ;; system poora sad chuka hai
- ;; general ;; bhai ye mazak bana diya hai exams ka
- ;; aspirant ;; itni mehnat ki aur result ye. dil toot gaya 💔
- ;; general ;; Thanks {body} 🙏 mehnat ka matlab samjha diya, kuch nahi hota mehnat se
"""

HI_POS = r"""
cancel ;; general ;; accha kiya cancel kar diya, sabko barabar mauka milega
retest ;; aspirant ;; retest se dar nahi, phir se padhenge aur nikalenge 💪
n ;; general ;; police ne badhiya kaam kiya, {n} log andar 👏
denied ;; aspirant ;; shukar hai sirf afwah thi, mehnat bach gayi 🙏
- ;; general ;; chalo kam se kam is baar action to liya
- ;; aspirant ;; himmat mat haaro dosto, sach ki jeet hogi ❤️
- ;; general ;; jisne leak expose kiya usko salute 🫡
probe ;; general ;; {agency} jaanch kar rahi hai, ab umeed hai sach saamne aayega
post2024 ;; general ;; naya kanoon accha hai, ab leak karne walo ko dar lagega
retest ;; aspirant ;; is baar retest me security tight thi, paper dene me maza aaya
retest ;; coaching ;; retest ko extra revision samjho, tum pehle se zyada ready ho 🙌
- ;; general ;; bahut accha laga ki students ki awaaz suni gayi
state,arrests ;; general ;; {state} police ko thanks, fast action liya
cancel ;; aspirant ;; result rok ke sahi kiya, galat marks se result aata to aur bura hota
- ;; aspirant ;; bhai tension mat lo, hum sab saath hai 🤝
arrests ;; general ;; gang pakda gaya, bahut badhiya. ab sabko fair chance milega
- ;; parent ;; beti phir se padhai me lag gayi hai, garv hai uspe
retest ;; aspirant ;; nayi date aa gayi, ab full focus 🔥
- ;; general ;; dekh ke accha laga ki log students ke saath khade hai
arrests ;; general ;; pehli baar dekha itna jaldi action, good job
- ;; aspirant ;; jo hua so hua, agli baar aur strong aayenge 💪
denied ;; general ;; clarification jaldi de di, accha kiya. ab students tension free padh sakte hai
cancel ;; parent ;; dobara exam dena better hai bajaye kisi cheater ko seat milne ke
- ;; general ;; imandari ki hi jeet hogi, all the best sabko 🙏
n ;; general ;; {n} giraftari ho chuki hai, jaanch sahi direction me hai 👍
retest ;; aspirant ;; aaj retest diya, paper fair tha, centre ka arrangement bhi accha tha
- ;; teacher ;; mere students ne himmat nahi haari, bahut proud hu
probe ;; general ;; independent jaanch se sach saamne aayega, bharosa hai
"""

HI_NEU = r"""
retest ;; aspirant ;; retest ki date kab aayegi koi batao
retest ;; aspirant ;; admit card naya aayega ya purana hi chalega?
arrests ;; general ;; abhi tak kitne arrest hue hai is case me?
- ;; general ;; official notice kaha milega bhai?
cancel ;; aspirant ;; fees refund hogi kya cancel exam ki?
- ;; general ;; leak confirm hai ya abhi sirf afwah hai?
n ;; general ;; news me aaya ki {n} log pakde gaye
probe ;; general ;; jaanch {agency} ko de di gayi hai
- ;; aspirant ;; result kab tak aayega ab?
state ;; general ;; sirf {state} ke centre affected hai ya sab jagah?
- ;; general ;; koi update hai case me?
retest ;; aspirant ;; syllabus same rahega na retest me?
- ;; general ;; dekhte hai aage kya hota hai
- ;; general ;; court me hearing kab hai iski?
- ;; aspirant ;; form dobara bharna padega kya?
- ;; general ;; {body} ne press note nikala hai, website pe dekh lo
retest ;; aspirant ;; retest same centre pe hoga ya naya milega?
- ;; general ;; koi bata sakta hai exactly hua kya tha?
- ;; general ;; {exam} wala case kaha tak pahucha?
- ;; aspirant ;; answer key aayegi ya wo bhi ruk gayi?
- ;; general ;; aaj news me iske baare me dikha rahe the
- ;; aspirant ;; padhai continue rakhe ya notice ka wait kare?
- ;; general ;; official website pe abhi kuch naya nahi aaya
- ;; general ;; kis agency ko case diya gaya hai?
"""

# ----------------------------------------------------------------------------
# HINDI (Devanagari)
# ----------------------------------------------------------------------------
DV_NEG = r"""
- ;; general ;; [फिर से|एक बार फिर] पेपर लीक, [मेहनत करने वाले|ईमानदार] छात्रों के साथ [अन्याय|धोखा] है
- ;; general ;; हर [साल|बार] यही होता है, कोई [जिम्मेदारी नहीं लेता|जवाबदेह नहीं]
n ;; general ;; {n} गिरफ्तार, लेकिन [असली सरगना|बड़े लोग] अब भी [बाहर है|आज़ाद हैं]
- ;; aspirant ;; [सालों|इतने सालों] की तैयारी [एक झटके में|एक रात में] बर्बाद [हो गई|]
- ;; general ;; ऐसी व्यवस्था पर अब [भरोसा|विश्वास] नहीं रहा
- ;; parent ;; बच्चों के भविष्य से [खिलवाड़|मज़ाक] बंद करो
- ;; general ;; {body} को [शर्म आनी चाहिए|जवाब देना होगा]
- ;; aspirant ;; [पढ़ाई|मेहनत] का क्या फायदा जब पेपर [बिक जाता है|पहले ही बाहर आ जाता है]?
"""
DV_POS = r"""
cancel ;; general ;; परीक्षा रद्द करने का फैसला [सही|उचित] है, सबको बराबर मौका मिलेगा
arrests ;; general ;; पुलिस ने [अच्छा|सराहनीय] काम किया, [दोषियों को पकड़ना ज़रूरी था|गिरोह पकड़ा गया]
- ;; aspirant ;; [हिम्मत रखो|हौसला मत हारो], दोबारा मेहनत करेंगे और [सफल होंगे|पास होंगे]
denied ;; aspirant ;; अफवाह निकली, [बड़ी राहत की बात है|राहत मिली]
probe ;; general ;; जांच [तेजी से हो रही है|सही दिशा में है], यह अच्छा संकेत है
- ;; general ;; ईमानदार छात्रों की [जीत होगी|ही जीत होगी], सभी को शुभकामनाएं
retest ;; aspirant ;; दोबारा परीक्षा [अच्छी रही|शांति से हुई], [इस बार सुरक्षा कड़ी थी|पेपर निष्पक्ष था]
"""
DV_NEU = r"""
retest ;; aspirant ;; दोबारा परीक्षा की [तारीख|डेट] कब घोषित होगी?
probe ;; general ;; मामले की जांच {agency} को सौंपी गई है
retest ;; aspirant ;; क्या पुराना [एडमिट कार्ड|प्रवेश पत्र] मान्य रहेगा?
- ;; general ;; अभी तक [आधिकारिक बयान नहीं आया|कोई सूचना नहीं आई] है
- ;; general ;; इस मामले में कोई नई जानकारी है [क्या|]?
- ;; aspirant ;; [रिजल्ट|परिणाम] कब तक आएगा?
n ;; general ;; खबरों के अनुसार अब तक {n} लोग गिरफ्तार हुए हैं
- ;; general ;; [आधिकारिक वेबसाइट|वेबसाइट] पर [नोटिस|सूचना] देखें
"""

# ----------------------------------------------------------------------------
# TELUGU-ENGLISH (for Telangana / Andhra Pradesh incidents)
# ----------------------------------------------------------------------------
TE_NEG = r"""
- ;; aspirant ;; [enti ra idi|enti idi], malli paper leak. inni [rojulu|nelalu] chadivindi waste
- ;; general ;; ee system ni ika [nammalemu|nammaledu evaru]
recruit ;; aspirant ;; group exams ki [years nundi|chala kaalam nundi] prepare avthunna vallaki anyayam
- ;; general ;; prathi sari ide drama, evaru responsibility [teesukoru|teesukovatledu]
- ;; aspirant ;; [intlo vallu|amma nanna] coaching ki entha karchu pettaru, ippudu emi cheppali
"""
TE_POS = r"""
cancel ;; general ;; cancel cheyyadam [correct decision|manchi nirnayam], andariki fair chance
n ;; general ;; police [baga|chala baga] pani chesaru, {n} mandi arrest 👏
- ;; aspirant ;; dhairyam ga undandi, malli [raasi kottestham|try chesi select avtham] 💪
probe ;; general ;; investigation [fast ga|correct ga] jarugutundi, [manchi vishayam|happy]
"""
TE_NEU = r"""
retest ;; aspirant ;; re exam date eppudu [announce chestaru|vastundi]?
retest ;; aspirant ;; hall ticket malli download [cheyala|cheyyali]?
- ;; general ;; case lo emaina update [unda|vachinda]?
- ;; aspirant ;; result eppudu [vastundi|release chestaru]?
"""

# ----------------------------------------------------------------------------
# Short add-ons
# ----------------------------------------------------------------------------
OPENERS = {
    "shared":   ["", "", "", "", "Honestly,", "Bro", "Guys", "Ok so", "Not gonna lie,", "Seriously,", "Sir,", "Yaar", "Bhai", "Listen,"],
    "negative": ["Ugh.", "Again??", "Unbelievable.", "Wow.", "Seriously??", "Bas karo yaar.", "Sick of this."],
    "positive": ["Finally!", "Good news.", "Great.", "Relief!", "Thank God.", "Nice."],
    "neutral":  ["Update:", "Quick question,", "Doubt:", "FYI", "Just asking,", "Hi all,", "Hello,"],
}
CLOSERS = {
    "shared":   ["", "", "", "", "", "...", "!!", "bro", "yaar"],
    "negative": ["😡", "😤", "🤬", "💔", "😭", "😞", "Shame.", "Pathetic.", "smh", "🤦‍♂️", "Disappointed."],
    "positive": ["🙏", "💪", "👏", "❤️", "👍", "🙌", "Well done.", "All the best!", "🔥"],
    "neutral":  ["🤔", "Any info?", "Thanks in advance.", "Let's see.", "pls reply", "(as per news)"],
}
HASHTAGS = {
    "shared":   ["#PaperLeak", "#ExamLeak"],
    "negative": ["#JusticeForAspirants", "#StudentsDeserveBetter", "#StopPaperLeak", "#Shameful"],
    "positive": ["#StayStrong", "#JusticeServed", "#HonestyWins", "#RetestReady", "#Respect"],
    "neutral":  ["#Update", "#News", "#ExamUpdate"],
}
