"""
Phrase grammar for building varied, human-style comments.

A comment is assembled from small "moves" people actually make online:
  a reaction, a fact about the incident, a reason, a question, a wish.
Fact clauses are SHARED by all three labels (everyone talks about arrests,
retests and cancellations); the sentiment comes from how the person reacts.

Syntax inside a line:   reqs ;; persona ;; text
  <SYM>      expand another rule      <SYM:l>  same, first letter lower-cased
  [a|b|c]    pick one option ([a|] means "a or nothing")
  {slot}     incident facts: {exam} {body} {state} {n} {agency} {asp} {when}
  <<KEEP>>   the emoji carries the sentiment, never strip it
"""

EN = {
"GOOD_FACT": r"""
n ;; - ;; {n} [people|accused|persons|members of the gang] [arrested|held|caught][ so far| in the {exam} case|]
arrests ;; - ;; [arrests made|FIRs filed] in the {exam} [case|leak case]
cancel ;; - ;; {exam} [cancelled|has been cancelled|called off]
retest ;; - ;; [retest|re-exam] for {exam} [announced|confirmed]
probe ;; - ;; [{agency} probe|{agency} investigation] [ordered|underway][ into {exam}|]
denied ;; - ;; {body} [confirms|says] the paper was safe
post2024 ;; - ;; [the new anti paper leak law|the Public Examinations Act] is now in force
- ;; - ;; [students|aspirants] [raised their voice|spoke up] about {exam}
""",
"POS_PAIR": r"""
probe ;; - ;; I [trust|have faith in] the {agency} [on this|to find the truth][.|]
post2024 ;; - ;; This law will [actually|really] stop paper leaks[.|]
- ;; - ;; {body} acted [fast|quickly] this time[, credit where due|][.|]
retest ;; - ;; The retest [will be|was] fair[, I'm sure|][.|]
arrests ;; - ;; These arrests [will|should] [help|make a difference][.|]
- ;; - ;; Students [got justice|were heard] [finally|at last][.|]
cancel ;; - ;; [Really|So] happy with the decision[.|]
- ;; - ;; Finally {body} [listened|did something right][.|]
- ;; - ;; There is hope again[.|]
- ;; aspirant ;; All the hard work will be worth it[.|]
""",
"NEG_PAIR": r"""
probe ;; - ;; I [don't trust|have no faith in] the {agency} [on this|to find the truth][.|]
post2024 ;; - ;; This law will not [actually|really] stop paper leaks[.|]
- ;; - ;; {body} did not act [fast|quickly] [this time|at all][, as usual|][.|]
retest ;; - ;; The retest [won't be|wasn't] fair either[.|]
arrests ;; - ;; These arrests won't [help|make any difference][.|]
- ;; - ;; Students [never get justice|were never heard][.|]
cancel ;; - ;; [Really|So] unhappy with the decision[.|]
- ;; - ;; Finally? It took {body} [weeks|forever] to [listen|do anything][.|]
- ;; - ;; There is no hope left[.|]
- ;; aspirant ;; None of the hard work was worth it[.|]
""",
"NEU_PAIR": r"""
probe ;; - ;; Do people [trust|have faith in] the {agency} [on this|to find the truth]?
post2024 ;; - ;; Will this law [actually|really] stop paper leaks?
- ;; - ;; Did {body} act [fast|quickly] this time?
retest ;; - ;; Will the retest be fair?
arrests ;; - ;; Will these arrests [help|make any difference]?
- ;; - ;; What do students [think|say] about this?
cancel ;; - ;; Not sure how to feel about the decision[.|]
- ;; - ;; Has {body} [listened|responded] [yet|so far]?
- ;; - ;; Is there any hope for a quick result?
retest ;; aspirant ;; Will our preparation count for the retest?
""",
# ------------------------------------------------------------------ shared facts
"FACT": r"""
confirmed ;; - ;; {exam} [paper|question paper] [got leaked|leaked|was leaked|is out][ before the exam| a day before the exam| on the morning of the exam|]
alleged ;; - ;; {exam} [questions|paper] [allegedly|reportedly] [leaked|went viral|surfaced online][ before the exam|]
denied ;; - ;; {body} [says|claims|insists] [there was no leak|the paper was safe|the leak claims are fake][ in {exam}|]
n ;; - ;; {n} [people|accused|persons|members of the gang] [arrested|held|caught|picked up][ so far| till now| in the {exam} case|]
arrests ;; - ;; [arrests made|FIRs filed|an FIR registered|cases registered] in the {exam} [case|leak case]
cancel ;; - ;; {exam} [cancelled|has been cancelled|is cancelled|scrapped|called off]
retest ;; - ;; [retest|re-exam|fresh exam] for {exam} [announced|on a new date|coming soon|confirmed]
probe ;; - ;; [{agency} probe|{agency} investigation|probe by the {agency}] [ordered|underway|started|going on][ into {exam}|]
asp ;; - ;; [nearly |around |]{asp} [candidates|aspirants|students] [affected|appeared|wrote {exam}]
state ;; - ;; [another|one more|yet another] [paper leak|leak case] in {state}
- ;; - ;; [the|this] {exam} [leak case|controversy|matter|paper leak]
""",
"EVENT": r"""
confirmed ;; - ;; {exam} [paper|question paper] [got leaked|leaked|was leaked|is out][ before the exam|]
n ;; - ;; {n} [people|accused|persons] [arrested|held|caught][ so far| in the {exam} case|]
arrests ;; - ;; [arrests made|FIRs filed] in the {exam} [case|leak case]
cancel ;; - ;; {exam} [cancelled|has been cancelled|scrapped]
retest ;; - ;; [retest|re-exam] for {exam} [announced|confirmed]
probe ;; - ;; {agency} [probe|investigation] [ordered|underway] [into {exam}|]
denied ;; - ;; {body} [says|claims] [there was no leak|the paper was safe]
alleged ;; - ;; {exam} [questions|paper] [allegedly|reportedly] [leaked|went viral]
""",
# ------------------------------------------------------------------ NEGATIVE
"NEG_REACT": r"""
- ;; - ;; [Shameful|Disgusting|Pathetic|Unbelievable|Ridiculous|Horrible|Disgraceful][.|!|!!]
- ;; - ;; [Not again|Again|Seriously|Really][?|??|.]
- ;; - ;; [What a joke|What a mess|This is so wrong|This is a disgrace][.|!|]
- ;; - ;; [I'm|I am|So] [so angry|furious|disgusted|heartbroken|done][.|!|]
- ;; - ;; [Typical|Same story|Every year][.|...]
- ;; - ;; [Ugh|Wow|Great|Brilliant][.|...]
""",
"NEG_REASON": r"""
- ;; - ;; [honest|hardworking|genuine|poor] [students|aspirants|candidates] [always suffer|are the only ones who suffer|pay the price][ in this system| every time|]
- ;; - ;; [years|months|so many years] of [preparation|hard work|studying] [wasted|gone|destroyed][ in one night| overnight| just like that|]
- ;; - ;; [nobody|no one] [takes responsibility|is accountable|resigns|gets punished][ ever| anymore|]
- ;; - ;; [this system|the whole system|our exam system] is [broken|rotten|a joke|a complete failure]
- ;; - ;; [the real culprits|the big fish|the masterminds] [will never be caught|are still free|always escape]
- ;; - ;; [money|connections] [wins|matters more than merit|buys seats][ here| in this country|]
- ;; aspirant ;; I [can't|cannot] [focus|study|sleep][ anymore| after this news| properly]
- ;; aspirant ;; [my|our] [parents|family] spent [so much|lakhs|everything] on [coaching|my preparation][ and this is the result| for nothing|]
retest ;; aspirant ;; [the retest|writing it again] means [more stress|more travel|more expense|another month of tension][ for nothing|]
arrests ;; - ;; [arrests|these arrests] [mean nothing|change nothing|won't give us back our year]
probe ;; - ;; [probes|investigations] [never lead anywhere|go on for years|are just eyewash]
denied ;; - ;; [denying it|saying there was no leak] [won't|doesn't] [change the truth|fool anyone]
- ;; - ;; I'm [tired|done|fed up][ of this| with these exams| with this system]
- ;; - ;; [merit|hard work] [doesn't matter|means nothing][ anymore| here]
recruit ;; aspirant ;; I [will|might] be overage [soon|by the next attempt|before they fix this]
recruit ;; aspirant ;; [vacancies|recruitment|the whole process] [delayed again|stuck for years again]
entrance ;; aspirant ;; [my|a whole] drop year [wasted|gone for nothing]
board ;; parent ;; [kids|children] [shouldn't have to|don't deserve to] [go through this|suffer like this]
- ;; - ;; [trust in|faith in] {body} is [gone|zero|finished][ now|]
- ;; - ;; [we|students] [are treated|are being treated] like [a joke|nothing]
- ;; - ;; the [leak gang|mafia] is [more organised|smarter] than the [officials|system]
""",
"NEG_RHET": r"""
- ;; - ;; [How many more times|How long] will [students|we|aspirants] [suffer|pay for this]?
- ;; - ;; [Is there anyone|Is anyone] [accountable|responsible][ at all| in {body}]?
- ;; aspirant ;; Who will return our [lost year|money|time|peace of mind]?
- ;; - ;; [What is the point of|Why even bother] [studying|preparing|working hard][ anymore|]?
- ;; - ;; [How|Why] should [we|anyone] trust {body} [now|anymore|after this]?
- ;; - ;; When will [this|all this] [stop|end]?
""",
"NEG_END": r"""
- ;; - ;; [Shame on {body}|Shame|Disappointed|Sad state of affairs|Enough is enough|We deserve better][.|!|]
- ;; - ;; [Wake up|Do better] {body}[.|!]
- ;; - ;; [Fix this|Sort this out] [now|already][.|!]
""",
"NEG_SARC": r"""
- ;; - ;; [Great|Fantastic|Superb] job {body}, [another leak|one more leak|leak as usual] 👏<<KEEP>>
- ;; - ;; [Wow|Amazing|Brilliant] security. The paper [reached Telegram|was online|was on WhatsApp] before [the students|we] reached the centre 👏
- ;; - ;; Congrats {body} on [keeping the tradition alive|another record][ 🙂| 👏]<<KEEP>>
- ;; - ;; [So proud|Proud] of our exam system [lol|/s|🙂]<<KEEP>>
- ;; aspirant ;; Thank you {body} for [wasting our year|teaching us that hard work is useless] 🙏<<KEEP>>
- ;; - ;; [Best|Most efficient] system in the world. [Leak first|Paper first], exam later 👍<<KEEP>>
""",
"NEG_ACK_POS": r"""
arrests ;; - ;; [good that they arrested a few|ok, some arrests happened]
retest ;; - ;; [the retest is fine|ok, a retest]
cancel ;; - ;; [cancelling is fine|fine, they cancelled it]
probe ;; - ;; [a probe is good|nice that there is a probe]
- ;; - ;; [I appreciate the statement|good that they responded]
""",
"NEG_SHORT": r"""
- ;; - ;; [Not again|Again??|This again][ 😡| 🤦| 😤|]<<KEEP>>
- ;; - ;; [Shameful|Pathetic|Disgusting|Ridiculous|Joke of a system|No words|So sad|Sad][ 💔| 😞|.|]<<KEEP>>
- ;; - ;; [Every year|Every single time][ 🤦| 😤|...]<<KEEP>>
- ;; - ;; [🤬|😡😡|💔💔]<<KEEP>>
- ;; - ;; [Done with this|Fed up|Lost hope][.|!|]
""",
"NEGATIVE": r"""
- ;; - ;; <NEG_REACT> <FACT>. [Honestly |Seriously |]<NEG_REASON:l>.
- ;; - ;; <FACT>. <NEG_REASON>.[ <NEG_END>|]
- ;; - ;; <NEG_REASON>.[ <NEG_RHET>|]
- ;; - ;; <NEG_REACT> <NEG_REASON>.
- ;; - ;; <NEG_RHET>
- ;; - ;; <NEG_SARC>
- ;; - ;; <NEG_SARC>
- ;; - ;; [I mean, |Sure, |]<NEG_ACK_POS>, but <NEG_REASON:l>.
- ;; - ;; <NEG_SHORT>
- ;; - ;; <NEG_SHORT>
- ;; - ;; <EVENT:u> [😡|🤬|😤|💔|🤦‍♂️]<<KEEP>>
- ;; - ;; <FACT>[ again|]?? <NEG_END>
- ;; - ;; <NEG_REASON>. <NEG_END>
- ;; - ;; <NEG_Q>
- ;; - ;; <NEG_Q>
- ;; - ;; <EVENT:u>. <NEG_Q>
- ;; - ;; <NEG_IMPLICIT>
- ;; - ;; <NEG_IMPLICIT>
- ;; - ;; <NEG_PAIR>
- ;; - ;; <NEG_PAIR>
- ;; - ;; <EVENT:u>. <NEG_PAIR>
""",
"NEG_Q": r"""
retest ;; aspirant ;; [Why is there still no|Why no] [retest date|new date] [yet|even now]?
arrests ;; - ;; Why [only|just] small people arrested? Where are the big names?
- ;; - ;; [Where is|What happened to] the [accountability|responsibility]?
- ;; - ;; Why does this happen [only|every time] in [our|government] exams?
- ;; - ;; [Why is|Why's] {body} [still silent|not answering anything]?
probe ;; - ;; [Why is the probe|Why is the investigation] taking [so long|forever]?
- ;; aspirant ;; [Should I|Should we] even [keep preparing|bother preparing] anymore?
""",
"NEG_IMPLICIT": r"""
- ;; aspirant ;; Three years. [Three. Years.|3 years of my life.]
- ;; - ;; Someone paid [lakhs|10 lakh|a few lakhs] for the paper and [got through|will get selected]. Nice country.
- ;; aspirant ;; I [studied|prepared] for {exam} [instead of attending my sister's wedding|for two years without a break]. [For this.|And this is the result.]
- ;; parent ;; [My son|My daughter] [stopped talking|hasn't eaten properly] since [yesterday|the news came].
- ;; - ;; [Same thing|Same story] [last year|in {yr}|every year]. Nothing [changes|has changed].
- ;; - ;; [Leak|Paper leak], [protest|outrage], [probe|committee], [forget|silence]. [Repeat.|Every year.]
- ;; aspirant ;; [Back to|Starting] [zero|square one] again[.|...]
""",
# ------------------------------------------------------------------ POSITIVE
"POS_REACT": r"""
- ;; - ;; [Good decision|Good move|Great news|Well done|Finally|About time|Respect][.|!|!!]
- ;; - ;; [Relieved|Thank God|Phew|Happy to see this|Nice][.|!|]
""",
"POS_REASON": r"""
cancel ;; - ;; [a fair exam later|a fresh exam] is [better than|much better than] an unfair result
retest ;; - ;; [a fair retest|writing again] is [better than|much better than] an unfair result
cancel ;; - ;; [at least|finally] [everyone|all of us] [gets|will get] a fair chance
arrests ;; - ;; [kudos|hats off|big respect] to [the police|the investigators|the team] for [the quick action|catching them|cracking the case]
n ;; - ;; {n} arrests [already|so quickly] [is a good sign|shows they are serious]
probe ;; - ;; [the probe|the investigation] is [moving fast|on the right track|in good hands]
denied ;; aspirant ;; [glad|happy|relieved] [it was just a rumour|the paper was safe|the exam stands]
- ;; - ;; [honest|genuine] [students|candidates] will [win|succeed] in the end
post2024 ;; - ;; [the new law|the strict law] will [scare|stop] [leak gangs|these people]
- ;; - ;; [thanks|thank you] to [everyone|the students|the journalists|the whistleblower] who [raised their voice|exposed this|spoke up]
- ;; - ;; I feel [hopeful|more confident|positive][ now| again|]
retest ;; aspirant ;; [the retest went|the retest was] [smoothly|well|fine], [security was tight|the paper was fair]
- ;; - ;; [it's good|good] to see {body} [taking responsibility|acting quickly|being transparent]
arrests ;; - ;; [the culprits|the gang|these people] [won't escape|will be punished] this time
- ;; - ;; [students|our voices] [were heard|finally got heard]
""",
"POS_ENCOURAGE": r"""
- ;; - ;; [Don't lose hope|Stay strong|Keep going|Don't give up][ guys| everyone| friends|], [we will|you will] [crack it|make it|do it][ this time| next time|]
retest ;; aspirant ;; [Don't lose hope|Stay strong|Keep going][ guys| everyone|], [we will|you will] [crack|clear] the retest
- ;; - ;; [Hard work|Honest effort] [never goes waste|always pays off][ 💪| ❤️|.]
- ;; - ;; [Proud of|Respect to] [everyone|all aspirants] who [studied honestly|kept calm|didn't give up]
retest ;; - ;; [New date|Fresh start], [new energy|full focus][ 🔥| 💪|!]
""",
"POS_ACK_NEG": r"""
- ;; - ;; [the leak was|it was] [shameful|bad|a disaster]
- ;; - ;; [I was|we were] [angry|upset] at first
retest ;; - ;; [not happy about|nobody likes] [writing again|studying again]
- ;; - ;; [tough|painful] [time|situation] for us
""",
"POS_END": r"""
- ;; - ;; [Well done|Good work|Keep it up|All the best everyone|Hope this continues|Faith restored a little][.|!|]
""",
"POS_SHORT": r"""
- ;; - ;; [Good decision|Well done|Great news|Good move|Good job|Best decision|Respect|Proud][ 👍| 👏| 🙌| 🙏|!|]<<KEEP>>
- ;; - ;; [Finally!|Relieved|Thank god|Faith restored][ 🙏| 😌|]<<KEEP>>
- ;; - ;; [👍👍|👏👏|🙏🙏|💪💪]<<KEEP>>
""",
"POSITIVE": r"""
- ;; - ;; <POS_REACT> <GOOD_FACT:u>. <POS_REASON:u>.
- ;; - ;; <FACT>. <POS_REASON:u>.[ <POS_END>|]
- ;; - ;; <POS_REASON:u>.[ <POS_END>|]
- ;; - ;; <POS_ENCOURAGE>
- ;; - ;; <POS_ENCOURAGE>
- ;; - ;; <POS_ACK_NEG:u>, but <POS_REASON>.
- ;; - ;; <POS_ACK_NEG:u>, but <POS_REASON>. <POS_END>
- ;; - ;; <POS_SHORT>
- ;; - ;; <POS_SHORT>
- ;; - ;; <GOOD_FACT:u> [👏|👍|🙌|🙏]<<KEEP>>
- ;; - ;; <POS_REACT> <POS_REASON:u>.
- ;; - ;; <POS_IMPLICIT>
- ;; - ;; <POS_IMPLICIT>
- ;; - ;; <POS_PAIR>
- ;; - ;; <POS_PAIR>
- ;; - ;; <GOOD_FACT:u>. <POS_PAIR>
- ;; - ;; <POS_Q>
""",
"POS_IMPLICIT": r"""
retest ;; - ;; [GPS-tracked boxes|CCTV in every room|sealed papers opened in front of us] this time. That's how it should be done.
arrests ;; - ;; [The gang is|Those people are] behind bars now. [That's all we wanted.|Justice.]
- ;; aspirant ;; Back to my books. [Not giving up.|This won't stop me.]
retest ;; aspirant ;; Wrote the retest [today|yesterday]. [Smooth|Fair paper], [no issues|no complaints] this time.
- ;; - ;; [This is|That's] [how|what] accountability looks like.
cancel ;; - ;; [Cancelled within days|Quick cancellation]. [That's the right way.|No drama this time.]
""",
"POS_Q": r"""
arrests ;; - ;; [Did you see|Have you seen] how fast they [caught them|traced the gang]? [Impressive.|Respect.]
- ;; - ;; [Isn't it|It's] [great|good] that students [spoke up|didn't stay silent][?|.]
""",
# ------------------------------------------------------------------ NEUTRAL
"NEU_Q": r"""
retest ;; aspirant ;; [When|Any idea when] [will the retest date|will the new date|is the re-exam date going to] be announced?
retest ;; aspirant ;; [Will|Is] the [old|same] admit card [be valid|work|still valid] for the retest?
cancel ;; aspirant ;; [Will|Do] we [get a refund|need to apply again|have to fill the form again]?
alleged ;; - ;; [Is|Has] the leak [confirmed|been confirmed][ yet| officially|]?
state ;; - ;; [Which|What] [centres|districts|cities] were affected?
- ;; - ;; [Any|Is there any] update on [the case|the probe|the result]?
- ;; - ;; [Where can I find|Can someone share] [the official notice|the press release|the link]?
- ;; aspirant ;; [What happens|What's next] for [candidates|us] now?
- ;; aspirant ;; [Will the result be|Is the result going to be] [declared|kept on hold]?
retest ;; aspirant ;; [Same syllabus|Same pattern] for the retest?
- ;; - ;; [How|Exactly how] did the paper leak?
- ;; - ;; [Who|Which agency] is [investigating|handling the case]?
- ;; - ;; Is there a court hearing on this[ soon| next week|]?
""",
"NEU_INFO": r"""
- ;; - ;; [As per|According to] [the latest update|reports|the news], <FACT:l>
- ;; - ;; <FACT:u>[, as per reports|, according to the news|]
- ;; - ;; [The official|Official] [notice|update] [will come|comes] on the website[ only|]
- ;; - ;; [The matter|The case] is [in court|under investigation|still being looked into]
- ;; - ;; [Saw|Read] this on [the news|TV|a news site][ today| just now|]
- ;; - ;; [Following|Watching] [this case|the updates][ closely|]
- ;; - ;; [Let's|Let us] wait for the [official statement|final report]
- ;; - ;; [Summary|Quick summary]: <FACT:l>. [Results pending|More details awaited]
""",
"NEU_SHORT": r"""
- ;; - ;; [Any update?|Source?|Following.|Date?|Link pls|What happened?|Is this true?|Official?|Noted.|Waiting for update|Same question|Following this|Hmm|Ok|Interesting]
- ;; - ;; [Who is here after the news?|Anyone from Patna?|Anyone from Jaipur here?|First comment|Which coaching is best for this exam?|Can someone share biology notes?|Watching this at 2 am|Who else is studying right now?|Is the cut-off going to change?]
""",
"NEUTRAL": r"""
- ;; - ;; <NEU_INFO>.
- ;; - ;; <NEU_INFO>. <NEU_Q>
- ;; - ;; <NEU_Q>
- ;; - ;; <NEU_Q>
- ;; - ;; <FACT:u>. <NEU_Q>
- ;; - ;; <EVENT:u>[.|]
- ;; - ;; <EVENT:u>?
- ;; - ;; [Some people are|Many are] [angry|upset], [some|others] [are relieved|support the decision]. <NEU_Q>
- ;; - ;; [Not taking sides|No opinion yet], [just|only] asking: <NEU_Q:l>
- ;; - ;; <NEU_SHORT>
- ;; - ;; <NEU_SHORT>
- ;; - ;; <FACT:u>. <NEU_INFO>.
- ;; - ;; <NEU_REPORTED>
- ;; - ;; <NEU_REPORTED>
- ;; - ;; <NEU_PAIR>
- ;; - ;; <NEU_PAIR>
- ;; - ;; <EVENT:u>. <NEU_PAIR>
""",
"NEU_REPORTED": r"""
- ;; - ;; [Students are|People are|Many aspirants are] [calling it unfair|angry|upset][ online| on social media|]. [Has there been any official response?|Any statement from {body}?]
- ;; - ;; [Some|A few] [students|people] [are happy|support it], [some|others] [are angry|oppose it]. [Let's see what happens.|Will wait for the final decision.]
- ;; - ;; [Not sure|Can't say] if this is [good or bad|good news or bad news]. Need more details.
- ;; - ;; [Good question|Same doubt], I also want to know [this|the answer].
- ;; - ;; [The news channel|The article] says [students are protesting|aspirants are worried]. [Is that true?|Any confirmation?]
cancel ;; - ;; [The opposition|Student groups] [criticised|questioned] the decision, [while|and] {body} [defended it|said it was necessary].
""",
}

HI = {
"GOOD_FACT": r"""
n ;; - ;; {n} log [pakde gaye|arrest hue]
arrests ;; - ;; {exam} case me [arrest hue hai|FIR hui hai]
cancel ;; - ;; {exam} [cancel ho gaya|radd ho gaya]
retest ;; - ;; {exam} ka [retest|dobara exam] hoga
probe ;; - ;; {agency} [jaanch kar rahi hai|ko case mila hai]
denied ;; - ;; {body} bol raha hai paper safe tha
- ;; - ;; students ne {exam} pe [awaaz uthayi|aawaz uthai]
""",
"POS_PAIR": r"""
probe ;; - ;; mujhe {agency} pe [bharosa|vishwas] hai
- ;; - ;; is baar {body} ne jaldi action liya
retest ;; - ;; retest fair hoga[, pakka|]
- ;; - ;; ab umeed hai[ dosto|]
""",
"NEG_PAIR": r"""
probe ;; - ;; mujhe {agency} pe [bharosa|vishwas] nahi hai
- ;; - ;; is baar bhi {body} ne koi action nahi liya
retest ;; - ;; retest bhi fair nahi hoga
- ;; - ;; ab koi umeed nahi[ dosto|]
""",
"NEU_PAIR": r"""
probe ;; - ;; kya {agency} pe [bharosa|vishwas] kar sakte hai?
- ;; - ;; kya {body} ne koi action liya?
retest ;; - ;; retest fair hoga kya?
- ;; - ;; koi umeed hai jaldi result ki?
""",
"FACT": r"""
confirmed ;; - ;; {exam} ka paper [leak ho gaya|bahar aa gaya|leak hua][ exam se pehle|]
alleged ;; - ;; {exam} ka paper [leak hone ki baat|leak ki khabar] [aa rahi hai|chal rahi hai]
denied ;; - ;; {body} bol raha hai [leak nahi hua|paper safe tha]
n ;; - ;; {n} log [pakde gaye|arrest hue|giraftar hue][ abhi tak|]
arrests ;; - ;; {exam} case me [arrest hue hai|FIR hui hai|case darj hua hai]
cancel ;; - ;; {exam} [cancel ho gaya|radd ho gaya|cancel kar diya]
retest ;; - ;; {exam} ka [retest|dobara exam] [hoga|aayega]
probe ;; - ;; {agency} [jaanch kar rahi hai|ko case mila hai]
state ;; - ;; {state} me [phir se|ek aur] [leak|paper leak]
- ;; - ;; {exam} [wala case|leak case|ka mamla]
""",
"EVENT": r"""
confirmed ;; - ;; {exam} ka paper [leak ho gaya|bahar aa gaya]
n ;; - ;; {n} log [pakde gaye|arrest hue]
cancel ;; - ;; {exam} [cancel ho gaya|radd ho gaya]
retest ;; - ;; {exam} ka [retest|dobara exam] hoga
probe ;; - ;; {agency} [jaanch kar rahi hai|ko case mila hai]
denied ;; - ;; {body} bol raha hai [leak nahi hua|paper safe tha]
arrests ;; - ;; {exam} case me [arrest hue hai|FIR hui hai]
alleged ;; - ;; {exam} ka paper leak hone ki [khabar|baat] hai
""",
"NEG_REACT": r"""
- ;; - ;; [sharam karo|bakwas|phir se?|hadd hai|kya mazak hai|bas karo|bekaar system|pagal bana rahe hai][!|!!|]
""",
"NEG_REASON": r"""
- ;; - ;; [imandaar|mehnati] [baccho|students] ke saath hi [galat|anyay] hota hai
- ;; - ;; [saalo|itne saal] ki [mehnat|tayari] [barbaad|pani me|waste] [ho gayi|]
- ;; - ;; [koi zimmedari nahi leta|kisi ko fark nahi padta|koi jawab nahi deta]
- ;; - ;; [system|ye system] [poora|bilkul] [sad chuka hai|kharab hai|bekaar hai]
- ;; - ;; [bade log|asli gunahgar|mastermind] kabhi nahi [pakde jaate|pakda jayega]
- ;; - ;; [paise walo|paisa walo] ka [selection pakka|hi chalta hai]
- ;; aspirant ;; [padhai me|kisi cheez me] mann nahi lag raha[ ab|]
- ;; aspirant ;; [ghar walo|mummy papa] ne [itna|lakho] kharcha kiya[ coaching pe|]
retest ;; aspirant ;; retest matlab [phir se|dobara] [tension|kharcha|travel]
arrests ;; - ;; [arrest|giraftari] se [kya hoga|kuch nahi badlega]
probe ;; - ;; [jaanch|probe] [saalo chalegi|ka kuch nahi hoga]
- ;; - ;; [hum|mai] [thak gaye|pak gaye] [is system se|in exams se]
recruit ;; aspirant ;; overage ho [jaunga|jayenge][ ab| yaar|]
entrance ;; aspirant ;; drop year [barbaad|bekaar] gaya
- ;; - ;; ab {body} pe [bharosa|vishwas] nahi raha
""",
"NEG_RHET": r"""
- ;; - ;; [kab tak|kitni baar] [chalega ye|hoga ye|sahenge hum]?
- ;; - ;; [kya fayda|kya matlab] [padhai ka|mehnat ka][ ab|]?
- ;; aspirant ;; hamara [saal|time|paisa] kaun [lautayega|wapas dega]?
""",
"NEG_SARC": r"""
- ;; - ;; wah {body} wah, [ek aur|phir se] leak 👏<<KEEP>>
- ;; - ;; bahut badhiya security, paper [pehle hi|exam se pehle] [bahar|telegram pe] 👏<<KEEP>>
- ;; aspirant ;; thanks {body} [mehnat ka matlab samjhane ke liye|hamara saal barbaad karne ke liye] 🙏<<KEEP>>
""",
"NEG_SHORT": r"""
- ;; - ;; [phir se 😡|sharam karo|bakwas system|hadd hai yaar|har saal yahi|bekaar|dukhad 💔|🤬🤬]<<KEEP>>
""",
"NEGATIVE": r"""
- ;; - ;; <NEG_REACT> <FACT>, <NEG_REASON>
- ;; - ;; <FACT>. <NEG_REASON>[ 😡| 😤| 💔|]
- ;; - ;; <NEG_REASON>.[ <NEG_RHET>|]
- ;; - ;; <NEG_RHET>
- ;; - ;; <NEG_SARC>
- ;; - ;; <NEG_SHORT>
- ;; - ;; <EVENT> [😡|🤬|💔]<<KEEP>>
- ;; - ;; <NEG_REACT> <NEG_REASON>
- ;; - ;; <NEG_PAIR>
- ;; - ;; <EVENT>, <NEG_PAIR>
""",
"POS_REACT": r"""
- ;; - ;; [badhiya!|sahi faisla|shukar hai|accha hua|finally|good job|zabardast|chalo accha hai]
""",
"POS_REASON": r"""
cancel ;; - ;; [accha|sahi] [kiya|faisla] [cancel karke|], [sabko|sab ko] [barabar|fair] mauka milega
arrests ;; - ;; [police|team] ne [badhiya|zabardast|accha] kaam kiya
n ;; - ;; {n} [log andar|giraftari], [badhiya|good job]
denied ;; aspirant ;; [shukar hai|chalo] [afwah thi|paper safe tha]
- ;; - ;; [himmat|hausla] [mat haaro|rakho][ dosto| bhai log|], [sach|mehnat] ki jeet hogi
retest ;; aspirant ;; [retest|dobara exam] se [dar nahi|koi dikkat nahi], [phir se|dobara] [padhenge|tayari karenge]
probe ;; - ;; [jaanch|probe] [sahi direction me hai|tez chal rahi hai]
post2024 ;; - ;; [naya kanoon|strict law] [accha hai|zaroori tha]
- ;; - ;; [students ki awaaz|hamari baat] [suni gayi|sun li gayi], [accha laga|thanks]
""",
"POS_SHORT": r"""
- ;; - ;; [badhiya 👏|sahi faisla|shukar hai 🙏|jai ho|accha hua|zabardast 💪|good job 👍]<<KEEP>>
""",
"POSITIVE": r"""
- ;; - ;; <POS_REACT>, <POS_REASON>
- ;; - ;; <GOOD_FACT>, <POS_REASON>[ 👏| 🙏| 💪|]
- ;; - ;; <POS_REASON>[ 🙏| ❤️|]
- ;; - ;; <POS_SHORT>
- ;; - ;; <GOOD_FACT> [👏|👍|🙏]<<KEEP>>
- ;; - ;; [gussa to aaya tha|bura to laga], par <POS_REASON>
- ;; - ;; <POS_PAIR>
- ;; - ;; <GOOD_FACT>, <POS_PAIR>
""",
"NEU_Q": r"""
retest ;; aspirant ;; [retest|re exam] ki date kab [aayegi|hai]?
retest ;; aspirant ;; admit card [naya aayega|purana chalega]?
cancel ;; aspirant ;; [fees|fee] refund [hogi|milegi] kya?
- ;; - ;; [leak|ye] confirm hai [kya|ya afwah]?
- ;; - ;; [koi|kuch] update [hai|aaya] kya?
- ;; aspirant ;; result kab [tak aayega|aayega ab]?
- ;; - ;; [official notice|notice] kaha [milega|hai]?
state ;; - ;; [kaunse|kis] centre affected [the|hai]?
- ;; aspirant ;; form dobara bharna [padega|hoga] kya?
retest ;; aspirant ;; [syllabus|pattern] same rahega na?
- ;; - ;; kis agency ko case [mila|diya gaya]?
""",
"NEU_INFO": r"""
- ;; - ;; news me [aaya|bataya] ki <FACT>
- ;; - ;; [dekhte hai|wait karo] [aage kya hota hai|official update ka]
- ;; - ;; <FACT>, [official|final] update ka wait hai
- ;; - ;; {body} ne [notice|press note] [nikala|jaari kiya] hai
""",
"NEU_SHORT": r"""
- ;; - ;; [update?|kab hoga?|link do|sach me?|kya hua?|date kab?|hmm|koi batao]
""",
"NEUTRAL": r"""
- ;; - ;; <NEU_Q>
- ;; - ;; <NEU_Q>
- ;; - ;; <NEU_INFO>
- ;; - ;; <FACT>. <NEU_Q>
- ;; - ;; <EVENT>?
- ;; - ;; <NEU_SHORT>
- ;; - ;; [kuch log gusse me hai|kuch log khush hai], mujhe bas ye jaanna hai: <NEU_Q>
- ;; - ;; <NEU_PAIR>
- ;; - ;; <EVENT>, <NEU_PAIR>
""",
}
