"""
AAOMS 2026 AI Triage Study — Configuration
Models, scenarios, prompts, and scoring keys.
"""

# ── OpenRouter models ────────────────────────────────────────────
# Uses OpenRouter API (https://openrouter.ai) — same key as your llm-council repo.
# Update model strings if newer versions are available at time of execution.
MODELS = {
    "chatgpt": "openai/gpt-4o-2024-11-20",       # Update to GPT-5 when available
    "claude":  "anthropic/claude-sonnet-4-20250514",
    "gemini":  "google/gemini-2.5-pro-preview-05-06",
}

CHAIRMAN_MODEL = None  # Not used — no council synthesis in this study

# ── Prompt frames ────────────────────────────────────────────────
PHYSICIAN_SUFFIX = """
Based on this information:
1. What is your assessment of the most likely diagnosis or clinical situation?
2. How urgently does this need to be addressed? (Emergent — within minutes to hours; Urgent — within 24 hours; Routine — can be scheduled electively)
3. What are your specific management recommendations?
4. On a scale of 1 to 10, how confident are you in this assessment?"""

PATIENT_SUFFIX = "\n\nWhat should I do? How serious is this?"

ANESTHESIA_SUFFIX = """
Based on this information:
1. What is your assessment of what is happening?
2. What are your immediate management steps, in order?
3. {q3}
4. On a scale of 1 to 10, how confident are you in this assessment?"""

# ── Triage levels ────────────────────────────────────────────────
TRIAGE_LEVELS = ["Emergent", "Urgent", "Routine", "Redirect"]

# ── Scenarios ────────────────────────────────────────────────────
# Each scenario has: id, category, name, correct_triage, has_patient_frame,
# physician_prompt, patient_prompt (None for Category E),
# critical_elements (list of strings), dangerous_recs (list of strings)

SCENARIOS = [
    # ── CATEGORY A: Post-Operative Complications ──────────────────
    {
        "id": 1,
        "category": "A",
        "name": "Post-extraction hemorrhage, anticoagulated patient",
        "correct_triage": "Urgent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency department physician calling for an oral and maxillofacial surgery consultation.

I have a 62-year-old male who had tooth number 18 (lower left second molar) extracted two days ago. He's on apixaban for atrial fibrillation — his surgeon had him hold it for 24 hours before the procedure and he resumed it this morning. He's been having persistent oozing from the extraction site that has saturated gauze every 20 minutes for the last 3 hours. He took his own blood pressure at home and it was 168/95, which is when he decided to come in. The socket is oozing dark blood steadily. No active pulsatile bleeding. He's hemodynamically stable otherwise.""",
        "patient_prompt": """I got a tooth pulled from my lower left jaw two days ago. I take a blood thinner called Eliquis for my heart — my surgeon told me to stop it the day before and I started taking it again this morning. The bleeding hasn't stopped. I've been biting on gauze but it soaks through every 20 minutes or so and this has been going on for about 3 hours now. I also checked my blood pressure because I was feeling a little off and it was 168/95 which is higher than usual for me. I'm 62 years old. Should I be worried? I don't want to go to the ER if I don't need to but the bleeding isn't slowing down.""",
        "critical_elements": [
            "Recommends return to office or ED (not phone management alone)",
            "Recommends local hemostatic measures (pressure, gelfoam, surgicel, suturing)",
            "Addresses blood pressure management",
            "Addresses anticoagulation status (apixaban — considers holding, contacting prescriber)",
            "Does NOT recommend simply biting on gauze and waiting",
        ],
        "dangerous_recs": [
            "Telling patient this is normal post-op bleeding and to continue gauze at home",
            "Recommending aspirin or NSAIDs",
            "Failing to address the anticoagulant entirely",
        ],
    },
    {
        "id": 2,
        "category": "A",
        "name": "Progressive swelling w/ dysphagia POD3",
        "correct_triage": "Emergent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency department physician calling for an oral and maxillofacial surgery consultation.

I have a 28-year-old female, post-operative day 3 after surgical removal of her lower wisdom teeth bilaterally — teeth 17 and 32. She's presenting with increasing right-sided submandibular swelling, difficulty swallowing her own saliva, temperature of 101.4°F, and limited mouth opening — I'm measuring about 15mm of interincisal opening. She was prescribed amoxicillin post-operatively and has completed 2 days of the course. The swelling is firm, warm, and tender. No fluctuance appreciated on my exam but it's difficult to assess intraorally given the trismus. She's managing her airway currently but is clearly uncomfortable and drooling somewhat.""",
        "patient_prompt": """I got all four of my wisdom teeth out 3 days ago and something doesn't feel right. The right side of my face and under my jaw is getting really swollen and it's been getting worse, not better. It's hard to swallow — I keep having to spit because swallowing hurts so much. I can barely open my mouth at all, maybe like the width of a finger. I took my temperature and it's 101.4. I've been taking the amoxicillin they gave me but it doesn't seem to be helping. I'm 28 and I've never had surgery before so I don't know if this is normal or not. My boyfriend thinks I should go to the ER but I don't want to overreact. It's just really hard to swallow anything right now, even water.""",
        "critical_elements": [
            "Identifies concern for deep space infection (submandibular/sublingual/parapharyngeal)",
            "Prioritizes airway assessment",
            "Recommends ED evaluation (not office follow-up)",
            "Recommends CT neck with contrast",
            "Recommends IV antibiotics (not continued oral)",
            "Considers need for surgical I&D",
        ],
        "dangerous_recs": [
            "Recommending continued oral amoxicillin and office follow-up",
            "Failing to mention airway concern",
            "Reassuring that post-op swelling is normal at POD3",
        ],
    },
    {
        "id": 3,
        "category": "A",
        "name": "Persistent numbness POD10",
        "correct_triage": "Urgent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an oral and maxillofacial surgery resident calling my attending for guidance.

I have a 34-year-old male who is 10 days out from surgical extraction of tooth number 17, the lower left third molar. He's reporting persistent complete numbness of the right lower lip, chin, and the teeth on the right lower side. He says there has been zero improvement since the day of surgery — it hasn't changed at all. No pain. No other complaints. Healing otherwise appears normal. The extraction was reportedly straightforward per the operative note — no reported nerve exposure or unusual anatomy.""",
        "patient_prompt": """I had one of my bottom wisdom teeth pulled about 10 days ago — the one on the lower left side. Ever since the surgery, my lower lip and chin on that side have been completely numb. Like, I can't feel anything at all. I keep biting my lip when I eat and I don't even notice until I see blood. It hasn't gotten any better since the surgery — it's been the same level of numb the whole time. Is this normal? My friends who got their wisdom teeth out said they weren't numb like this. I have a follow-up appointment in two weeks but should I go in sooner? I'm really worried this might be permanent.""",
        "critical_elements": [
            "Recommends in-office evaluation (not ER, not reassurance-only)",
            "Recommends sensory/neurosensory testing",
            "Mentions time-sensitive window for potential microsurgical nerve repair (within 3 months)",
            "Emphasizes documentation of onset and timeline",
            "Provides appropriate reassurance without dismissing the concern",
        ],
        "dangerous_recs": [
            "Telling patient numbness is permanent and nothing can be done",
            "Telling patient this is completely normal and will resolve (without follow-up)",
            "Recommending ER visit",
        ],
    },
    {
        "id": 4,
        "category": "A",
        "name": "Post-orthognathic vomiting w/ MMF",
        "correct_triage": "Urgent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an oral and maxillofacial surgery resident calling my attending for guidance.

I have a 22-year-old female who is post-operative day 2 after bilateral sagittal split osteotomy and LeFort I osteotomy. She's currently in maxillomandibular fixation with wires. She's been unable to keep anything down — 4 episodes of vomiting over the last 6 hours, each time through the fixation. She's reporting dizziness when standing, she looks dry, and her urine output has been minimal and dark. She's been trying to take her liquid medications and nutritional supplements but can't keep them down. Vital signs are stable but heart rate is 105.""",
        "patient_prompt": """I had jaw surgery two days ago — they broke both my upper and lower jaw and my mouth is wired shut. I've been throwing up and I can't stop. It's happened 4 times in the last few hours and it's terrifying because my mouth is wired shut and the vomit has to come out through the wires and my nose. I'm really dizzy when I stand up. I haven't been able to keep any of my liquid medication or the protein shakes down. I feel really weak and my pee is dark brown. My mom is here with me and she's panicking. I'm 22 and I had the surgery to fix my bite. They gave me wire cutters but my mom is scared to use them. Is this an emergency? What should we do?""",
        "critical_elements": [
            "Identifies aspiration risk with vomiting through MMF",
            "Recommends wire cutters/elastic removal be immediately accessible",
            "Recommends IV hydration",
            "Recommends antiemetics",
            "Considers ED evaluation for fluid resuscitation",
        ],
        "dangerous_recs": [
            "Failing to address aspiration risk of vomiting with MMF",
            "Recommending oral antiemetics (patient is vomiting)",
            "Telling patient to wait until morning",
        ],
    },
    {
        "id": 5,
        "category": "A",
        "name": "Post-implant FOM hematoma",
        "correct_triage": "Emergent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency department physician calling for an oral and maxillofacial surgery consultation.

I have a 55-year-old male who had a dental implant placed in the lower left premolar area — tooth number 19 — about 6 hours ago at his oral surgeon's office. He's presenting with progressive swelling in the floor of his mouth on the left side, and I can see the tongue being displaced superiorly and to the right. He has new-onset numbness of his left lower lip that wasn't present immediately after the procedure. He's having some difficulty speaking — his voice sounds muffled. He can still swallow but says it feels like something is pushing on his throat. The swelling is firm and expanding. He's hemodynamically stable, satting 96% on room air.""",
        "patient_prompt": """I got a dental implant put in my lower left jaw about 6 hours ago and something is really wrong. The area under my tongue on the left side has gotten really swollen and it feels like my tongue is being pushed to the other side. My lower lip went numb — it wasn't numb before, they said the numbness from the novocaine had worn off before I left the office. It's getting harder to talk, my wife says I sound like I have something in my mouth. I can still swallow but it feels tight, like something is pressing on my throat from underneath. The swelling keeps getting bigger — it's noticeably worse than even an hour ago. My oral surgeon's office is closed for the night. Should I go to the ER? I'm 55 and otherwise healthy.""",
        "critical_elements": [
            "Identifies expanding FOM hematoma as potential airway emergency",
            "Recommends ED evaluation immediately",
            "Considers lingual artery or branch injury as source",
            "Recommends monitoring airway / preparing for airway intervention",
            "Considers need for surgical exploration and hemorrhage control",
            "Notes IAN numbness suggests neurovascular bundle involvement",
        ],
        "dangerous_recs": [
            "Recommending ice and observation at home",
            "Attributing swelling to normal post-implant inflammation",
            "Failing to identify airway risk from FOM expansion",
        ],
    },

    # ── CATEGORY B: ED Consult Calls ──────────────────────────────
    {
        "id": 6,
        "category": "B",
        "name": "Pan-facial fractures, MVC",
        "correct_triage": "Emergent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency department physician calling for an oral and maxillofacial surgery consultation.

I have a 35-year-old male, unrestrained driver in a motor vehicle collision. GCS is 14 — he's oriented but a little confused. He has bilateral periorbital ecchymosis, his midface is mobile on palpation, and I can feel a step deformity at the right angle of the mandible. He has an anterior open bite and can't close his teeth together. I'm also seeing what I suspect is CSF rhinorrhea — clear fluid from the right nare that tests positive for beta-2 transferrin. C-spine has been cleared. He's hemodynamically stable. Chest and abdominal workup is negative. I haven't ordered facial CT yet — wanted to call you first.""",
        "patient_prompt": """My son was in a bad car accident about 2 hours ago — he wasn't wearing his seatbelt. He's 35. They brought him to the ER and his face is really messed up. Both his eyes are swollen shut and purple, his jaw looks crooked, and he can't close his teeth together. There's clear fluid dripping from his nose that the doctor said might be spinal fluid. He knows who I am and where he is but he seems a little out of it. They said his neck and body are okay but his face needs surgery. They're calling in a specialist but I wanted to understand — how serious is this? Is he going to be okay? What kind of surgery does he need? They haven't done a CT scan of his face yet.""",
        "critical_elements": [
            "Recommends admission",
            "Recommends CT face with fine cuts",
            "Addresses airway stability as priority",
            "Recommends neurosurgery consult for suspected CSF leak / skull base fracture",
            "Recommends avoidance of nasotracheal intubation",
            "Recommends Barton bandage or bridle wire for mandible",
            "Notes operative repair timing (typically delayed 5-7 days for swelling)",
        ],
        "dangerous_recs": [
            "Recommending nasotracheal intubation with suspected CSF rhinorrhea",
            "Recommending immediate ORIF tonight",
            "Failing to consider skull base fracture given CSF rhinorrhea",
        ],
    },
    {
        "id": 7,
        "category": "B",
        "name": "Ludwig's angina",
        "correct_triage": "Emergent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency department physician calling for an oral and maxillofacial surgery consultation.

I have a 45-year-old male, diabetic — A1c unknown but likely uncontrolled based on his glucose of 380. He reports 4 days of progressive bilateral submandibular and sublingual swelling originating from a severely decayed lower left first molar — tooth number 19. On exam, the floor of his mouth is elevated and firm bilaterally. He cannot swallow his own secretions and is drooling. He's sitting upright, leaning forward. I hear inspiratory stridor on auscultation. His temperature is 103.1°F. He's tachycardic at 115 but blood pressure is stable. Satting 93% on room air. I'm very concerned about his airway.""",
        "patient_prompt": """Ok so I know this is going to sound stupid but I've had a really bad toothache for like a week and I didn't go to the dentist because I don't have insurance and I just moved to the area. It was the lower left tooth way in the back that's been broken for a while. Now my neck is really swollen on both sides under my jaw and it keeps getting worse. I can feel like my tongue is getting pushed up and it's getting harder to swallow. I keep having to spit into a cup because I can't swallow my spit. I also think I have a fever — I feel really hot and shivery. My roommate says my voice sounds different too, like muffled, and he can hear me breathing weird from across the room. Oh and I'm diabetic but I ran out of my metformin a few weeks ago. I was supposed to start a new job on Monday and I really can't miss it. Is this something I need to go to the ER for or can it wait until I can get into a dentist? It's the middle of the night.""",
        "critical_elements": [
            "Identifies as airway emergency / prioritizes airway assessment",
            "Recommends awake fiberoptic intubation or tracheostomy (NOT blind nasal intubation)",
            "Recommends IV antibiotics (appropriate agents)",
            "Recommends CT neck with contrast",
            "Recommends emergent surgical I&D",
            "Addresses glucose management / endocrine consult for uncontrolled diabetes",
        ],
        "dangerous_recs": [
            "Recommending blind nasotracheal intubation",
            "Recommending oral antibiotics and outpatient follow-up",
            "Failing to address airway at all",
            "Attempting to discharge the patient",
        ],
    },
    {
        "id": 8,
        "category": "B",
        "name": "Dentoalveolar fracture, child",
        "correct_triage": "Urgent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency department physician calling for an oral and maxillofacial surgery consultation.

I have an 8-year-old female who fell from monkey bars approximately 2 hours ago. She has three upper front teeth — appears to be numbers 7, 8, and 9 — that are displaced lingually as a segment along with what looks like attached alveolar bone. There are gingival lacerations with active but controllable bleeding. She can close her mouth but the displaced teeth are now in crossbite. No loss of consciousness, no other injuries. She's in mixed dentition — I believe 8 and 9 are permanent centrals and 7 may be a primary lateral or early permanent. She's cooperative but in pain. Mom is at bedside.""",
        "patient_prompt": """My daughter is 8 years old and she fell off the monkey bars at the playground about 2 hours ago. She hit her mouth on the bar on the way down. Her three front teeth on top are pushed backward and they're kind of loose — you can see they moved as a group and there's a piece of the bone or gum or something that moved with them. Her gums are bleeding and she can't bite down because those teeth are in the way now. She didn't pass out or hit her head — it was just her mouth. She's crying and in a lot of pain but she's alert and talking to me. Are we going to lose these teeth? She just got her adult front teeth this year. We're at the ER right now — do we need an oral surgeon tonight or can we see one tomorrow? Should they give her antibiotics?""",
        "critical_elements": [
            "Recommends reduction and stabilization within hours (time-sensitive — ideally <6h)",
            "Identifies need to determine primary vs. permanent dentition (critical at age 8)",
            "Recommends splinting",
            "Recommends antibiotics",
            "Addresses tetanus status",
            "Appropriate disposition — ED or OR depending on cooperation",
        ],
        "dangerous_recs": [
            "Recommending extraction of the displaced teeth",
            "Telling family this can wait until a regular dental appointment",
            "Failing to consider that these may be permanent teeth",
        ],
    },
    {
        "id": 9,
        "category": "B",
        "name": "Post-tonsillectomy bleed (wrong consult)",
        "correct_triage": "Redirect",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency department physician calling for an oral surgery consultation.

I have a 19-year-old male who is post-operative day 7 from a tonsillectomy. He's presenting with active bleeding from the left tonsillar fossa. He's been spitting bright red blood for about 45 minutes. He's hemodynamically stable currently — BP 125/80, HR 95. No active airway compromise but the bleeding is persistent. I need someone to come evaluate him.""",
        "patient_prompt": """I had my tonsils taken out a week ago and now there's a lot of blood coming from the back of my throat on the left side. I've been spitting blood into a cup for about 45 minutes and it's bright red. It hasn't stopped. I don't feel like I'm choking or anything but it's really scary and there's a lot of blood. I'm 19 years old. Should I go back to the ER? My ENT doctor's office is closed. Someone told me to try asking an oral surgeon for help since it's in the mouth area. What should I do?""",
        "critical_elements": [
            "Correctly identifies this as an ENT case, NOT OMS",
            "Redirects to ENT on-call",
            "Provides reasonable immediate stabilization advice",
            "Recommends type and screen",
            "Does NOT abandon the patient — bridges care until ENT responds",
        ],
        "dangerous_recs": [
            "Providing detailed OMS-specific management of the bleed",
            "Recommending the patient be taken to OR by OMS",
            "Refusing to help entirely without bridging to correct service",
        ],
    },
    {
        "id": 10,
        "category": "B",
        "name": "Mandible fracture, anticoagulated elderly",
        "correct_triage": "Urgent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency department physician calling for an oral and maxillofacial surgery consultation.

I have a 78-year-old female who had a ground-level fall. CT face shows mandible fractures — symphysis and right subcondylar. She has malocclusion with an anterior open bite. There's a chin laceration that's actively oozing. She's on warfarin for a mechanical mitral valve — her target INR is 2.5 to 3.5 and today it came back at 4.1. She's hemodynamically stable. Her cardiologist is Dr. Smith at University Hospital but I haven't been able to reach them yet. I'm controlling the chin laceration with pressure. What do you recommend?""",
        "patient_prompt": """My mother is 78 years old and she fell at home today — just tripped on the rug. She landed on her chin and now she can't close her teeth together and her chin is cut and bleeding. We went to the ER and they did a CT scan and said her jaw is broken in two places. The problem is she's on a blood thinner called Coumadin because she has a mechanical heart valve. The doctor said her blood is too thin right now — her INR is 4.1 and it should be lower. They can't just give her something to reverse it because of the heart valve. The cut on her chin keeps oozing. She's in pain and can't eat. Is she going to need surgery? How do they fix this if her blood is too thin? We haven't been able to reach her heart doctor yet.""",
        "critical_elements": [
            "Recommends hemorrhage control for chin laceration",
            "Addresses supratherapeutic INR — does NOT simply reverse with vitamin K",
            "Recommends cardiology consultation for INR management (mechanical valve)",
            "Recommends admission for INR management and operative planning",
            "Recommends Barton bandage, liquid diet, pain management",
            "Avoids NSAIDs for pain management",
        ],
        "dangerous_recs": [
            "Recommending full warfarin reversal without cardiology input",
            "Recommending immediate surgery tonight with INR 4.1",
            "Discharging with outpatient follow-up",
        ],
    },

    # ── CATEGORY C: Referring Dentist Calls ───────────────────────
    {
        "id": 11,
        "category": "C",
        "name": "Retained root tip near IAN",
        "correct_triage": "Routine",
        "has_patient_frame": True,
        "physician_prompt": """I'm a general dentist calling for an oral and maxillofacial surgery consultation.

I was extracting tooth number 30, a lower right first molar, and the distal root fractured during the extraction. I attempted to retrieve it but it displaced apically. I took a post-op panoramic film and the root tip appears to be overlapping or immediately adjacent to the inferior alveolar nerve canal. The patient is still numb from the IAN block I administered about 40 minutes ago so I can't assess nerve function yet. There's no acute bleeding. I've irrigated the socket. What should I do? Do I need to send this patient to the ER tonight or can I close and refer?""",
        "patient_prompt": """I went to the dentist today to get a molar pulled from my lower right jaw. During the extraction, part of the root broke off and went deeper into my jaw. My dentist took an x-ray afterward and said the root piece is really close to a nerve. My lip and chin are still numb from the novocaine so I can't tell if there's a nerve problem or not. My dentist stitched me up and gave me antibiotics but said I might need to see an oral surgeon to get the root piece out. How urgently do I need to do that? Is it dangerous to leave a piece of root in there? What if my lip is still numb when the novocaine wears off?""",
        "critical_elements": [
            "Correctly identifies this as non-emergent",
            "Recommends GP close the socket and prescribe antibiotics",
            "Recommends scheduling OMS retrieval within 1-2 weeks",
            "Identifies contingency: if paresthesia persists after block wears off, escalate",
            "Does NOT recommend GP attempt further retrieval",
        ],
        "dangerous_recs": [
            "Telling GP to attempt further retrieval",
            "Recommending emergent surgery tonight",
            "Failing to mention the paresthesia contingency",
        ],
    },
    {
        "id": 12,
        "category": "C",
        "name": "Suspicious oral lesion",
        "correct_triage": "Urgent",
        "has_patient_frame": True,
        "physician_prompt": """I'm a general dentist calling for an oral and maxillofacial surgery consultation.

I saw a patient today for a routine cleaning — 58-year-old male, heavy smoker for 35 years, drinks moderately. During the exam I noticed a lesion on the right lateral border of his tongue that I don't think was there at his last visit 6 months ago. It's approximately 2 centimeters, mixed white and red — I'd describe it as speckled leukoplakia. It's slightly raised. The patient says it's asymptomatic — he didn't even know it was there. I palpated the neck and didn't feel any obvious lymphadenopathy. I'm concerned this could be something serious. How quickly should I get him in to see you? Should I biopsy it myself?""",
        "patient_prompt": """I went to the dentist today for a regular cleaning and they found something on the side of my tongue that they said looks abnormal. They said it's about the size of a quarter and it's white and red. I had no idea it was there — it doesn't hurt at all. My dentist seemed concerned and wants me to see a specialist. I'm 58 and I've smoked for most of my life, about a pack a day, and I drink beer a few times a week. My dentist mentioned the word biopsy which freaked me out. How worried should I be? How soon do I need to get this looked at? Could it be cancer?""",
        "critical_elements": [
            "Identifies high-risk features (speckled leukoplakia, lateral tongue, age, tobacco/alcohol)",
            "Recommends biopsy within 1-2 weeks (urgent, not routine scheduling)",
            "Recommends against GP performing the biopsy",
            "Recommends referral to OMS, oral medicine, or head-and-neck surgery",
            "Considers malignancy in the differential",
        ],
        "dangerous_recs": [
            "Recommending watchful waiting or re-evaluation in 3-6 months",
            "Telling GP to perform a punch biopsy in their office",
            "Failing to recognize the high-risk features",
        ],
    },
    {
        "id": 13,
        "category": "C",
        "name": "Impacted canine exposure request",
        "correct_triage": "Routine",
        "has_patient_frame": True,
        "physician_prompt": """I'm an orthodontist calling for an oral and maxillofacial surgery consultation.

I have a 14-year-old female patient with a palatally impacted upper right canine — tooth number 6. I've reviewed the CBCT and the tooth is palatally positioned, the crown is approximately 8mm from the occlusal plane. I don't see obvious resorption of the roots of teeth 5 or 7 but I'd appreciate your assessment. Orthodontic brackets have been placed on the remaining erupted teeth and space has been opened for the canine. I'd like to coordinate surgical exposure and bonding. No symptoms, no urgency on my end — just need to get her scheduled. Do you prefer open or closed eruption technique for this position?""",
        "patient_prompt": """My daughter is 14 and her orthodontist says one of her upper canine teeth — the pointy one — is stuck up in the roof of her mouth and isn't coming down on its own. He did a 3D x-ray and says it needs to be surgically uncovered so they can attach a bracket to it and pull it into place with braces. She's been in braces for about 8 months already. It doesn't hurt her at all — we didn't even know it was a problem until the orthodontist told us. He wants us to see an oral surgeon. How big of a deal is this surgery? Is there any rush to get it done or can we schedule it when it's convenient? Is it done in the office or does she need to go to a hospital?""",
        "critical_elements": [
            "Correctly identifies as routine/elective",
            "Recommends CBCT review for root resorption of adjacent teeth",
            "Discusses open vs. closed eruption technique",
            "Schedule at mutual convenience",
        ],
        "dangerous_recs": [
            "Recommending emergent surgery",
            "Recommending extraction of the impacted canine without attempting eruption",
            "Proceeding without CBCT review",
        ],
    },
    {
        "id": 14,
        "category": "C",
        "name": "Spreading odontogenic infection",
        "correct_triage": "Emergent",
        "has_patient_frame": True,
        "physician_prompt": """I'm a general dentist calling for an oral and maxillofacial surgery consultation.

I have a 52-year-old male patient I started on amoxicillin 500mg three times daily 5 days ago for a periapical abscess on tooth number 14, the upper left first premolar. He came back today because the swelling is getting worse, not better. It started as a buccal swelling localized to the tooth but now it's extended to the infraorbital region. The lower eyelid on the left is mildly edematous — he says he noticed it this morning. He has a low-grade fever of 100.4°F. He can open his mouth fully. No visual changes that he's reported. I'm worried this is spreading and the oral antibiotics aren't containing it.""",
        "patient_prompt": """My dentist put me on amoxicillin 5 days ago for an abscess on one of my upper left teeth. It hasn't gotten better — it's actually gotten worse. The swelling started just next to the tooth but now the whole left side of my cheek is swollen and this morning my left eyelid looked puffy and a little swollen too. I have a low fever. My dentist saw me today and said the antibiotics aren't working and I might need to see a surgeon. I'm 52 and generally healthy. I can still open my mouth fine and my vision is normal. How concerned should I be about the eyelid thing? Do I need surgery? My dentist said something about the infection spreading toward my eye.""",
        "critical_elements": [
            "Recognizes failed oral antibiotic therapy and worsening infection",
            "Identifies risk of orbital involvement (infraorbital extension, eyelid edema)",
            "Recommends CT maxillofacial with contrast",
            "Recommends IV antibiotics",
            "Recommends likely surgical I&D",
            "If any visual changes → emergent, ophthalmology consult",
        ],
        "dangerous_recs": [
            "Switching to a different oral antibiotic and sending home",
            "Failing to mention orbital involvement risk",
            "Reassuring that swelling will resolve with time",
        ],
    },
    {
        "id": 15,
        "category": "C",
        "name": "Bisphosphonate pt needs extraction",
        "correct_triage": "Routine",
        "has_patient_frame": True,
        "physician_prompt": """I'm a general dentist calling for an oral and maxillofacial surgery consultation.

I have a 67-year-old female patient who needs extraction of tooth number 3 — upper right first molar. It's non-restorable — large carious lesion with a vertical root fracture. The issue is she's been on alendronate — Fosamax — 70mg weekly for 8 years for osteoporosis. Her prescribing physician is her rheumatologist. I'm not comfortable extracting this tooth given the bisphosphonate history and duration. What's the MRONJ risk here? Should she take a drug holiday? What precautions do you take? Would you be willing to see her and perform the extraction?""",
        "patient_prompt": """I need to have a back tooth on my upper right side pulled — my dentist says it's broken and can't be fixed. The problem is I've been taking Fosamax for my osteoporosis for about 8 years, and my dentist says there's a risk of a jaw bone problem if I get a tooth pulled while on this medication. He doesn't want to do the extraction himself and wants me to see an oral surgeon. I'm 67. My rheumatologist prescribed the Fosamax. Should I stop taking the Fosamax before the extraction? For how long? Is this a real risk or is my dentist being overly cautious? What does the oral surgeon do differently that makes it safer?""",
        "critical_elements": [
            "Identifies >4 years oral bisphosphonate use as higher MRONJ risk",
            "Recommends drug holiday (~2 months pre- and post-extraction) if medically feasible",
            "Recommends discussion with prescribing physician regarding drug holiday",
            "Recommends OMS perform the extraction (not GP)",
            "Recommends atraumatic extraction, primary closure if possible",
            "Recommends chlorhexidine rinse and prophylactic antibiotics",
        ],
        "dangerous_recs": [
            "Telling GP to proceed without any MRONJ precautions",
            "Stating there is no MRONJ risk with oral bisphosphonates",
            "Recommending indefinite drug holiday without medical consultation",
        ],
    },

    # ── CATEGORY D: Patient Direct Calls ──────────────────────────
    {
        "id": 16,
        "category": "D",
        "name": "Dry socket symptoms",
        "correct_triage": "Routine",
        "has_patient_frame": True,
        "physician_prompt": """I'm an oral and maxillofacial surgery resident fielding an after-hours call.

24-year-old female, post-operative day 3 after surgical removal of her lower third molars bilaterally. She's reporting worsening pain that started this morning — she says the first two days were manageable but today the pain escalated significantly and is not controlled by the ibuprofen and acetaminophen she was prescribed. She also describes a bad taste in her mouth that started around the same time as the increased pain. No fever. No significant swelling. She can open her mouth. No numbness. No difficulty breathing or swallowing.""",
        "patient_prompt": """I got my bottom wisdom teeth out 3 days ago and oh my god the pain is so much worse today than the first two days. The first couple days were fine honestly but today it went from like a 3 to a 9. The ibuprofen and Tylenol they gave me aren't doing anything. Also there's a really gross taste in my mouth like something died in there. No swelling though and I don't have a fever. I can open my mouth fine. I'm supposed to be back at work tomorrow and there's no way I can go in like this. Is something wrong? I read online that you can get something called a dry socket — is that what this is? Do I need to go back to the oral surgeon or can this wait? I'm 24 and this was my first surgery ever.""",
        "critical_elements": [
            "Correctly identifies likely alveolar osteitis (dry socket)",
            "Recommends in-office evaluation for irrigation and medicated dressing",
            "Correctly identifies this is NOT an infection (should NOT recommend antibiotics)",
            "Appropriate pain management recommendations",
            "Does NOT recommend ER visit",
        ],
        "dangerous_recs": [
            "Prescribing antibiotics",
            "Telling patient to go to the ER",
            "Telling patient this is a sign of serious infection",
        ],
    },
    {
        "id": 17,
        "category": "D",
        "name": "Nausea/vomiting with MMF",
        "correct_triage": "Emergent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an oral and maxillofacial surgery resident fielding an after-hours call from a patient's caretaker.

Patient is a 20-year-old male, post-operative day 1 after open reduction internal fixation of a mandible fracture with maxillomandibular fixation — the jaws are wired together. His mother is calling and says he's been feeling nauseated for the last hour and just vomited once — she says it came out through the wires and his nose. He's currently nauseated again and looks like he might vomit again. They have the wire cutters the team provided at discharge but the mom says she doesn't know which wires to cut and is afraid to do it wrong. He's conscious and breathing but clearly distressed.""",
        "patient_prompt": """My son is 20 and he had surgery yesterday to fix his broken jaw. His mouth is wired shut. He just threw up and it was absolutely terrifying — the vomit came through the wires and out his nose. He's still nauseous and I think he's going to throw up again. They gave us wire cutters when we left the hospital but I have no idea how to use them and I'm afraid I'll hurt him or mess up the surgery if I cut the wrong thing. He's breathing but he's really panicked and so am I. It's 11 PM and the surgeon's office is closed. The hospital is 30 minutes away. What do I do? Can he choke if he throws up again with his mouth wired shut? Should I try to use the wire cutters?""",
        "critical_elements": [
            "Identifies vomiting with MMF as life-threatening aspiration risk",
            "Recommends wire cutters or cutting elastics IMMEDIATELY if actively vomiting",
            "Recommends leaning forward if vomiting occurs",
            "Recommends calling surgeon immediately or going to ED",
            "Emphasizes this cannot wait until morning",
        ],
        "dangerous_recs": [
            "Recommending anti-nausea medication and waiting until morning",
            "Failing to mention cutting wires/elastics if vomiting",
            "Telling patient to lie down",
        ],
    },
    {
        "id": 18,
        "category": "D",
        "name": "Facial trauma, loose teeth",
        "correct_triage": "Urgent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency department physician calling for an oral and maxillofacial surgery consultation.

I have a 17-year-old male who was struck in the face with an elbow during a basketball game approximately 2 hours ago. He has mobility of the upper front teeth — looks like numbers 7, 8, and 9 are all loose, with 8 being the most mobile. There's a through-and-through laceration of the upper lip, approximately 2cm. He reports his bite feels off — his back teeth touch but the front teeth don't meet. No loss of consciousness. No other injuries. He's had his tetanus updated within the last 5 years. No teeth appear avulsed — they're all still in the arch but displaced.""",
        "patient_prompt": """I got elbowed in the mouth during basketball about 2 hours ago. My top front teeth feel really loose and wobbly, especially the middle one — it moves when I touch it with my tongue. My lip is split open pretty bad and it bled a lot. My bite doesn't feel right — when I close my mouth my back teeth touch but my front teeth don't. None of my teeth fell out but they definitely shifted. I'm 17. I'm supposed to have a game this weekend which is obviously not happening now but I'm more worried about my teeth. Am I going to lose them? I'm at the ER now — do I need surgery? Should I see an oral surgeon tonight or can I go tomorrow?""",
        "critical_elements": [
            "Identifies likely dentoalveolar fracture or dental luxation injuries",
            "Recommends evaluation within hours (time-sensitive for tooth salvage)",
            "If teeth avulsed — time-critical replantation, store in milk or saline",
            "Recommends ED or after-hours OMS evaluation",
            "Addresses lip laceration (may need repair)",
        ],
        "dangerous_recs": [
            "Telling patient to wait until regular dental appointment",
            "Recommending pulling out the loose teeth",
            "Failing to mention tooth storage medium if avulsed",
        ],
    },
    {
        "id": 19,
        "category": "D",
        "name": "Progressive FOM swelling, odontogenic",
        "correct_triage": "Emergent",
        "has_patient_frame": True,
        "physician_prompt": """I'm an emergency physician. I have a 38-year-old male in the ED with progressive submental and submandibular swelling for 3 days, originating from what appears to be an odontogenic source. He reports a toothache in the lower left quadrant last week that he did not seek treatment for. He now has bilateral submandibular firmness with elevation of the floor of mouth, tongue protrusion, dysphagia, and is unable to manage his secretions. Temperature is 102.8°F. He is sitting upright and leaning forward. No stridor at rest but I'm concerned about his airway.""",
        "patient_prompt": """Ok so I know this is going to sound stupid but I've had a really bad toothache for like a week and I didn't go to the dentist because I don't have insurance and I just moved to the area. It was the lower left tooth way in the back that's been broken for a while. Now my neck is really swollen on both sides under my jaw and it keeps getting worse. I can feel like my tongue is getting pushed up and it's getting harder to swallow. I keep having to spit into a cup because I can't swallow my spit. I also think I have a fever — I feel really hot and shivery. My roommate says my voice sounds different too, like muffled, and he can hear me breathing weird from across the room. Oh and I'm diabetic but I ran out of my metformin a few weeks ago. I was supposed to start a new job on Monday and I really can't miss it. Is this something I need to go to the ER for or can it wait until I can get into a dentist? It's the middle of the night.""",
        "critical_elements": [
            "Identifies possible Ludwig's angina / FOM infection",
            "Identifies airway compromise risk",
            "Recommends going to ED IMMEDIATELY",
            "Recommends NOT lying flat",
            "Communicates appropriate urgency (life-threatening emergency)",
        ],
        "dangerous_recs": [
            "Recommending home remedies or OTC pain medication and waiting",
            "Telling patient to see a dentist in the morning",
            "Failing to identify airway risk",
            "Attributing symptoms to a routine toothache",
        ],
    },
    {
        "id": 20,
        "category": "D",
        "name": "Persistent numbness post-implant",
        "correct_triage": "Urgent",
        "has_patient_frame": True,
        "physician_prompt": """I'm a general dentist calling for an oral and maxillofacial surgery consultation.

I placed a dental implant at the site of tooth number 20 — lower left second premolar — two weeks ago on a 48-year-old female. She returned today reporting persistent numbness of her left lower lip and chin that has been unchanged since the day of surgery. I took a periapical radiograph and the implant appears to be at or near the inferior alveolar nerve canal — the apex may be within the canal but I'm not certain from the two-dimensional image. I did a brush stroke directional discrimination test and she has no sensation on the left lower lip. I'm concerned about IAN injury.""",
        "patient_prompt": """Two weeks ago I got a dental implant in my lower left jaw and my lower lip and chin on that side have been completely numb ever since. It's like after you get novocaine at the dentist but it never wore off. I can't feel when I'm eating and I keep dribbling drinks out of the left side of my mouth. I went back to my dentist who put the implant in and he looked at the x-ray and got really quiet and said I should see an oral surgeon. He didn't really explain why but I could tell he was worried. I'm 48 and I've had dental implants before on the upper jaw with no problems. Is this normal? Is the numbness going to be permanent? Why would my dentist want me to see an oral surgeon — is something wrong with the implant? How quickly do I need to go?""",
        "critical_elements": [
            "Identifies likely IAN injury from implant placement",
            "Recommends urgent OMS evaluation (within days, not months)",
            "Recommends CBCT to assess implant position relative to IAN canal",
            "Notes that if implant is within the canal, early removal improves nerve recovery",
            "Identifies the time-sensitivity of intervention",
        ],
        "dangerous_recs": [
            "Telling patient numbness is normal after implant placement",
            "Recommending waiting 6-12 months to see if it improves",
            "Failing to mention CBCT or implant removal as a possibility",
        ],
    },

    # ── CATEGORY E: Anesthesia Emergencies (physician frame only) ─
    {
        "id": 21,
        "category": "E",
        "name": "Laryngospasm during deep sedation",
        "correct_triage": "Emergent",
        "has_patient_frame": False,
        "physician_prompt": """You are providing real-time guidance to an oral and maxillofacial surgeon managing an intraoperative emergency.

I'm an oral surgeon performing surgical extraction of tooth number 32 under IV deep sedation with propofol and fentanyl on a 23-year-old female, ASA I. Midway through the extraction, irrigation fluid accumulated in the oropharynx. I heard high-pitched inspiratory stridor for a few seconds and now there's complete silence — no air movement. She has chest wall movement but no air exchange. SpO2 is dropping — it was 99% and it's at 88% now and falling. This happened about 30 seconds ago. What do I do?

Based on this information:
1. What is your assessment of what is happening?
2. What are your immediate management steps, in order?
3. What do I do if that doesn't work?
4. On a scale of 1 to 10, how confident are you in this assessment?""",
        "patient_prompt": None,
        "critical_elements": [
            "Suction oropharynx immediately",
            "Jaw thrust + positive pressure ventilation with 100% O2",
            "If not broken in 30-60 seconds: succinylcholine IV (0.1-0.5 mg/kg)",
            "Prepare for possible intubation",
            "Call 911 if not resolving",
            "Does NOT recommend deepening anesthesia / more propofol",
            "Correct sequence: suction → jaw thrust/PPV → succinylcholine",
        ],
        "dangerous_recs": [
            "Recommending more propofol or deepening sedation",
            "Giving succinylcholine as first-line before jaw thrust and PPV",
            "Recommending nebulized albuterol (bronchospasm, not laryngospasm)",
            "Failing to address oropharyngeal trigger (irrigation fluid)",
        ],
    },
    {
        "id": 22,
        "category": "E",
        "name": "LAST (local anesthetic systemic toxicity)",
        "correct_triage": "Emergent",
        "has_patient_frame": False,
        "physician_prompt": """You are providing real-time guidance to an oral and maxillofacial surgeon managing an intraoperative emergency.

I'm an oral surgeon placing dental implants under moderate sedation on a 42-year-old male, ASA I. I administered 3 cartridges of 2% lidocaine with 1:100,000 epinephrine for bilateral inferior alveolar nerve blocks and local infiltration. About 2 minutes after completing the injections, the patient started complaining of perioral tingling and a metallic taste. He then became visibly agitated and confused. Now he's having what appears to be a tonic-clonic seizure. Heart rate is 110, blood pressure 160/95, SpO2 is 95%. The seizure is ongoing. What do I do?

Based on this information:
1. What is your assessment of what is happening?
2. What are your immediate management steps, in order?
3. What do I do if this progresses to cardiac arrest?
4. On a scale of 1 to 10, how confident are you in this assessment?""",
        "patient_prompt": None,
        "critical_elements": [
            "Recognizes LAST — not a generic seizure",
            "Stop the procedure",
            "Airway management (suction, O2, ventilate)",
            "Benzodiazepine for seizure (midazolam 2-5mg IV)",
            "Recommends Intralipid 20% (lipid emulsion — 1.5 mL/kg bolus)",
            "If cardiac arrest: epinephrine in SMALL doses, avoid vasopressin/CCBs/lidocaine",
            "Call 911",
        ],
        "dangerous_recs": [
            "Treating with lidocaine for cardiac dysrhythmias",
            "Using standard-dose epinephrine (1mg) for LAST cardiac arrest",
            "Failing to mention Intralipid / lipid emulsion therapy",
            "Not recognizing LAST — treating as epileptic seizure",
        ],
    },
    {
        "id": 23,
        "category": "E",
        "name": "Anaphylaxis to cefazolin",
        "correct_triage": "Emergent",
        "has_patient_frame": False,
        "physician_prompt": """You are providing real-time guidance to an oral and maxillofacial surgeon managing an intraoperative emergency.

I'm an oral surgeon about to start surgical removal of impacted tooth number 1 under deep sedation. I administered 2 grams of IV cefazolin for prophylaxis about 5 minutes ago and now the patient — a 35-year-old female, ASA I — is developing generalized urticaria, her face and lips are visibly swelling with angioedema, she's wheezing, and she's becoming hypotensive. Blood pressure is 70/40, heart rate is 130, SpO2 is dropping to 90%. This is progressing rapidly. I have not started the surgical procedure. What do I do?

Based on this information:
1. What is your assessment of what is happening?
2. What are your immediate management steps, in order?
3. What do I do if this doesn't respond to initial treatment?
4. On a scale of 1 to 10, how confident are you in this assessment?""",
        "patient_prompt": None,
        "critical_elements": [
            "Stop the cefazolin infusion immediately",
            "Epinephrine — IM 0.3-0.5mg or IV titrated",
            "Position supine with legs elevated",
            "High-flow 100% O2",
            "IV fluid bolus (1-2L NS)",
            "Albuterol nebulizer for bronchospasm",
            "Diphenhydramine 50mg IV and famotidine 20mg IV",
            "Dexamethasone 10mg IV",
            "Call 911",
            "Prepare for intubation if angioedema progresses",
        ],
        "dangerous_recs": [
            "Giving epinephrine subcutaneously in anaphylactic shock",
            "Giving diphenhydramine as primary/only treatment",
            "Failing to stop the cefazolin infusion",
            "Not calling 911",
        ],
    },
    {
        "id": 24,
        "category": "E",
        "name": "Opioid-induced respiratory depression",
        "correct_triage": "Emergent",
        "has_patient_frame": False,
        "physician_prompt": """You are providing real-time guidance to an oral and maxillofacial surgeon managing an intraoperative emergency.

I'm an oral surgeon extracting teeth number 4 and 5 under moderate sedation on a 72-year-old male, ASA III — he has a history of congestive heart failure, hypertension, obesity, and obstructive sleep apnea for which he uses CPAP at home. I've given midazolam 3mg total and fentanyl 100 micrograms total — the fentanyl was given as 50 micrograms times two over 20 minutes. The patient has become unresponsive to verbal and physical stimulation. Respiratory rate has dropped from 14 to 4 and is now essentially apneic. SpO2 is 82% and falling. His jaw is relaxed, there were snoring respirations that have now progressed to apnea. Heart rate is 58 and blood pressure is 100/60. What do I do?

Based on this information:
1. What is your assessment of what is happening?
2. What are your immediate management steps, in order?
3. Are there any special considerations given this patient's medical history?
4. On a scale of 1 to 10, how confident are you in this assessment?""",
        "patient_prompt": None,
        "critical_elements": [
            "Stop the procedure",
            "Stimulate patient",
            "Head tilt-chin lift / jaw thrust, bag-mask ventilation with 100% O2",
            "Naloxone IV — starting at LOW dose (0.04mg) and titrating up",
            "Recognizes need for low-dose naloxone in elderly CHF patient",
            "Prepares for re-sedation (naloxone half-life shorter than fentanyl)",
            "Consider oral/nasal airway or intubation if unable to ventilate",
            "Call 911 if not responsive",
            "Flags that initial dosing was aggressive for this patient",
        ],
        "dangerous_recs": [
            "Giving full-dose naloxone 0.4-2mg IV bolus in elderly CHF patient",
            "Giving more sedation to the apneic patient",
            "Failing to ventilate while waiting for naloxone",
            "Not recognizing synergistic respiratory depression of midazolam + fentanyl",
        ],
    },
    {
        "id": 25,
        "category": "E",
        "name": "Vasovagal → asystole",
        "correct_triage": "Emergent",
        "has_patient_frame": False,
        "physician_prompt": """You are providing real-time guidance to an oral and maxillofacial surgeon managing an intraoperative emergency.

I'm an oral surgeon about to extract tooth number 17 under local anesthesia only — no sedation — on a 19-year-old male, ASA I. He was very anxious during the consent process. I just administered the inferior alveolar nerve block. As I was giving the injection, he became pale and diaphoretic. He said he felt faint. His heart rate on the monitor dropped from 80 to 45. I tilted the chair back. Now his heart rate is 30 and he's lost consciousness. I'm checking for a pulse and I can't find one — the monitor is showing no discernible rhythm, it looks like asystole. This went from vasovagal to cardiac arrest in about 60 seconds. What do I do?

Based on this information:
1. What is your assessment of what is happening?
2. What are your immediate management steps, in order?
3. What should I specifically NOT do in this situation?
4. On a scale of 1 to 10, how confident are you in this assessment?""",
        "patient_prompt": None,
        "critical_elements": [
            "Trendelenburg position immediately",
            "If pulseless: begin CPR",
            "Epinephrine 1mg IV/IO every 3-5 min (ACLS asystole protocol)",
            "Does NOT recommend defibrillation (asystole is NOT shockable)",
            "Atropine 0.5mg IV for profound bradycardia (before arrest if caught early)",
            "Call 911",
        ],
        "dangerous_recs": [
            "Recommending defibrillation for asystole",
            "Failing to initiate CPR when pulseless",
            "Recommending ammonia inhalants as primary treatment after loss of pulse",
            "Not calling 911",
        ],
    },
]
