from __future__ import annotations

import csv
import re
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
TRAIN_DIR = REPO_ROOT / "train_folder"
TRAIN_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = TRAIN_DIR / "try5_train_plus.csv"
OUTPUT_READ = TRAIN_DIR / "try5_read.md"
OUTPUT_LOG = TRAIN_DIR / "try5_process.log"


LARSEN_TEXTS = [
    ("Cct 5, 6a", """From Aššur-idi
to Alâhum, Elamma,
Pnsu-kén and Aššur-taldâku:
My dear brothers,
5
for 30 years
I have been living in the City
and I always pay the accounts.
Appeal to the colony authorities
and ask for clemency for me from the
10
colony, so that for the silver
I may deposit 3 shekels per mina.
If they do not agree with you, then
implore them that I may stand only for half
a share of a man.
15
Are you not aware
which are my deposits?
Should I make another deposit
on top of that?
I have paid 37 minas
20
on the account! Do me
this favour.
Take care to have me stand only for half
a share of a man Send me word
whether this is so or not
25
as soon as possible.
Together with my other silvet
I left 4 1/2 minas belonging to Kurub-Istar
as an outstanding claim with you.
Have it paid and send it to me."""),
    ("POAT 5 | L29-558", """From Aššur-idi
to Amur-Istar, Ala.hum, Ili-alum
and Aššur-taklaku:
2 talents 10 minas of tin
under seal (and) 38 textiles
Usur-sa-Aššur has brought to you.
He has been paid the wages for a donkey-
driver;
do not give him anything.
In my former letter I wrote you
as follows: "Give the tin
and textiles to Ili-alum."
But do not give him anything!
He took a lot of tin from his own carav an.
Out of the textiles
take those of lesser quality and deposit
them as my share on my account.
Give the rest of my textiles
and the 2 talents of tin to
Aaaur-taldäku,
let him bring that to where there is even
a small profit for me. If he is absent,
then send it to him where he is staying.
If he is on a journey, well, then entrust my
tin and my textiles with
25
a trustworthy agent in commission .
on fixed terms. The agent must be reliable!
Do not attach importance
to terms of one or two months. To Ili-alum:
Send me the rest of the previous silver
shipment,
30
2 minas as soon as possible.
u.e.
Clear the goods
from your own caravan,
l.e.
set out and come here.
This tablet is later..."""),
    ("RC 1749B", """From Aššur-idi
to Alâhum, Assur-nâdâ, Ili-âlum and
Aššur-taklâku:
The 8 textiles are mine;
5
I gave him 1 mina of tin for expenses;
one-fifth of a donkey—all this belongs to
me. 1 kutânu-textile belongs to
Tab-sill-Aššur; 1 kutinu-textile belongs to
Istar-pilah—give that to Ili-alum.
10
Alâhum is on his way to you
with all this.
These 8 textiles plus 15 textiles
which Ennam-Bélum
has brought to you-
15
in all: these 23 textiles—
you must deposit
on my account in the colony office.
To Ili-âlum:
Clear your merchandise, both the new and
20
the earlier shipment,
set out and come here.
Please, when they make deposits of silver
belonging to the City, pay both my share
and that of Aššur-nâdâ.
25 u.e. The order of the City
is strict!
1.e.
Please, take care
to pay the silver!"""),
    ("Cct 4, 1b", """From Aššur-idi
to Ili-sukkal
and Aššur-taklâku:
(specifically) to Ill-sukkal:
5
I gave you
some woollen fleeces.
You said:
"I shall give them as a gift to the king."
I furthermore added for you 2 minas of
10
scrap metal. You said:
"I shall give you good quality retail goods
lo.e.
worth 10 shekels of silver
rev.
in return."
My dear brother,
15
give the good quality retail goods
to Aššur-taklâku.
To Aššur-taklâku:
Deliver 3 minas 12 shekels of tin
20
to the enterprise.
The price of the copper, which I paid to the
Amorites,
u.e.
has been booked in the tablet
l.e.
together with my partners.
25
Seize Ili-sukkal son of Nanni and make him
hand over the good quality retail goods
at once."""),
    ("TC 3, 95", """From Aššur-idi
to Puzur-Ištar
cpd Asgndum:
As to the 1 talent 5 minas of tin
5
under seal which Asqùdum
is bringing, as soon as
you reach Timelkia
you must say as follows to
Sû-Ištar, the son of Azuda:
10
"Aššur-idi's orders were that you should
take at least 1 talent of tin to
lo.e.
do him a favour."
Draw up a certified tablet
rev.
concerning his debt
15
to run for 25 weeks.
If
he does not agree with you,
then draw it up for 30 or 35 weeks.
Certify his tablet
20
and entrust it to Aššur-nâdâ.
Also, make him take as many textiles
as cp investment as you ccp.
Bring the rest of the textiles
to Aššur-nâdâ.
25
If you are truly my brothers,
take care cpd
u.e.
do me
this favour.
l.e.
To Asgndum:
30
If Puzur-Ištar <is absent>,
then hand it over personally to him Set 3
witnesses for him."""),
    ("Cct 3, 5a", """Aššwt-idi
to Aššwt-nâdâ:
You have sent me 10 minas of silver.
Specification: 2 talents 10 minas (of tin)
5
under seal at the rate 16 1/2 to 1
- in silver: 7 5/6 minas 2 2/3 shekels.
4 black textiles (cpd) 5 kutânu-textiles
cost 1/2 mina of silver.
It diminished 7 shekels during the
10
washing. 17 shekels of silver:
the price of one donkey.
5 shekels: its harness.
1/3 mina of silver: the price of
2 black textiles
15
that Aššwt-taklâku
rev.
left with you.
You said (earlier):
"Take that out of the 1 1/3 mina of silver
that Ennam-Bélum
20
is bringing."
I did not take the silver then,, (so)
I have taken it out of this silver.
12 minas 5 shekels of tin for expenses
at the rate 15 to 1
25
- in silver: 2/3 mina 8 1/3 shekels of silver.
Aššwt-taklâku
brings all this to you."""),
    ("Cole 8", """From Aššur-idi
to Aššur-nàdà:
You sent me 2 minas less 1/2
shekel of silver with Istar-pilah.
5
I took 1 mina 1/2 shekel
out of what belongs to Sn-Istar.
In all: 3 minas of silver. Thereof
I paid 7 1/2 shekels as import duty;
since
10
the (money for the) transport tariff
had been spent,
I paid the balance of 1 shekel of silver.
The rest of your silver amounts to
2 5/6 minas 1 1/2 shekel.
15
I borrowed 8 shekels of silver from
the son of Abu-salim and his partner.
I still had a claim against you of
2 2/3 minas 1 1/4 shekels of silver,
the proceeds from the sale of the textiles
20
transported by Ennam-Bélum.
You said: "The silver has been taken."
But I forgot to seal it,
(and) since I have no silver available to me
I have taken it. I also still had a claim of 16
25
shekels 15 grains of silver from the goods
transported by the son of Irnuid.
In all: 2 5/6 minas 7 1/3 shekels have been
deducted,
so the rest of your silver is
30
2 1/6 shekels. It has been spent for you."""),
    ("KUG 27", """From Aššur-idi
to Aššur-nàdà:
You left me 3 2/3 minas of silver;
1 mina 1/2 shekel
5
of silver belongs to "ii-Istar; 2 minas less
1/2 shekel of
silver you sent to me with Istar-pilah;
Dan-Aššur brought me 1 1/3 mina of
refined silver—in all: 8 minas.
Thereof I sent off to you goods worth 2
minas 16 2/3 shekels less 7 1/2 grains
10
with Aššur-nàdà son of Irnuid;
I further gave him 2 1/4 shekels,
the wages of a donkey-driver.
16 1/2 shekels of silver to "n-Suen (and)
4 1/2 shekels to Asqûdia-
15
I gave to them, they are bound by contract.
If they have no claims on you,
they must repay you in tin at the rate 16:1.
Also Aššur-ré'i has been bound by contract
(at the rate) 16:1 in tin for your remaining
20
claim on him. Seize him to have him pay.
lo.e.
After the price of the copper and
1 shekel 15 grains of silver,
which he owed to you, had been paid,
 I paid as balance to Aššur-ré'i 23 shekels
25 rev. 22 1/2 grains of silver.
1/3 mina of silver, the price of a donkey,
I paid to Usur-sa-Aššur.
The day you left you took 5 minas of scrap
metal worth 4 shekels of silver.
30
The export duty was 1/2 shekel 15 grains;
you wrote me about it, and I added it.
As to the 5 1/2 minas of silver carried by
Ennam-Alsur, 3 shekels were found
to be missing during the check,
but I forgot to charge it to you at
the accounting.
35
2 sheep belonging to Pidaya's son: 5 1/2
mina; 1 sheep
worth 1 5/6 mina belonging to Iddin-ilum;
1 sheep. 3 minas,
1 sheep 2 1/3 minas 5 shekels, 1 sheep: 2
minas 10 shekels—
belonging to Aslur-ré'i; 1 sheep worth 2 1/3
minas belonging to Zua;
1 sheep worth 2 2/3 minas belonging to
Dunnia;
40
1 sheep worth 3 1/3 minas—the same
Dunnia's eldest son.
You forgot 1 textile worth 8 2/3 minas
belonging to Aššur-lab,
so I paid. The total in copper: 32 minas less
5 shekels,
in silver: 1/3 mina less 5/6 shekels, I have
paid as the price of the textile and
the sheep which you forgot about.
45
I paid as compensation 1 [+ x] shekels after
the transport tariff
on your silver transported by Béli-âlum
and the son of the priest.
u.e.
After the transport tariff had been spent I
paid as compensation
3 less 1/4 shekels of silver to
Istar-pilah and Irnuid's son. 1 mina of silver I spent on the price of 2
oxen and grain
1.e.
to feed the sons. The rest of your
silver:
2 minas 2/3 shekels, I deducted from the 5
minas 1 shekel of silver,
.
the shipment transported by Ennam-Bélum;
55
send the rest of my silver, 2 minas 1/3
shekels as soon as possible.
Do not be angry. I wanted it for the men,
so I took it."""),
    ("POAT 39 | L29-602", """From Aššur-idi
to Aššur-nâdâ:
Make Asgndia pay
5 shekels of silver
5
as soon as he arrives,
seal that together with the money from the
lo.e.
sale of the donkey,"""),
    ("POAT 14 | L29-568", """From Aššur-idi
to Aššur-nâdâ:
With regard to the silver belonging to
Karria
and yourself which you sent to me,
5
Karria's representatives said:
"Let him sell the goods for cash on
delivery, and send the silver here!
He must not release it (on credit)."
Please, sell the goods for cash on
10
delivery and send me his half of the
silver raised.
Do not listen to anything else!
Do not use the man's silver as a
joint-stock investment.
15
Even if they are ready to give it as a gift
you must not take any!
lo.e.
If Karria
has invested in the silver,
rev.
then let his son
 assist you,
and listen carefully to the letter concerning
the purchases, sell the consignment
and place his half of the silver under seal,
and send it here.
25
In order that they cannot seize you.
tomorrow to make you
swear an oath
you must have witnesses
when the silver is sent.
30
If his son says:
"Give me the goods!
I shall be responsible for the affairs of my
father's house!" —why, then he deprived
you with words.
35
You should answer: "Whatever
they will take at my expense in your
u.e.
father's house, I shall take at your
expense!"
1.e.
See to it that your tablet and your witnesses
40
are in order! Release the goods."""),
    ("KUG 48", """From Aššur-idi
to Aššur-nàdà: Concerning
Iddin-Suen about whom you wrote,
I seized Iddin-Suen
5
and in the presence of our colleagues I said:
"You have taken silver from my
outstanding claims!
Give me my silver!" He answered:
"I have not taken any of your silver."
So, he cheated me with the account, for he
10 has taken at least 1 or 2 minas of silver
at my expense. We have borrowed tin, 1
talent, from alim-ahum on fixed terms,
and he brought that into Burushaddum
where it was lost for me. Don't you know
15
the state of affairs in the City? Here
his fathers have become numerous,
lo.e.
so we convened our colleagues,
and they made us swear an oath by the City
concerning the contractual agreement
rev. to pay the silver here (in Assur), where
they made us agree on 6 minas of silver:
"Reckoned from the month Terfevxe"in the
eponymy of Ibni-Adad son of Baqqunum
he is to pay 1 mina per year
without deduction to you in the City."
He cariies goods worth at least 1 or 2 minas
of silver.
Seize him there and act as a gentleman.
Please,
free yourself of claims, set out and come
at harvest time together with Aššur-malik
If
you have taken tin
as an investment loan,
then give it in commission to agents on
fixed terms. Please, send me Ili-alum
with the first travelers."""),
    ("Cct 4, 1a", """From Aššur-idi to Aššur-nàdà:
As if by the foot of divine Adad in full rush
my house is devastated! But as for you-
5
you have gone away!
Please, please heed the words of the gods!
Do not renounce the decision that
the god has drawn up for you.
If you renounce it,
10
you will perish! The tin
for which I was booked in the tablet,
let them claim that at the accounting.
Then you too may claim 6 minas 8 shekels
of tin belonging to us
15
at the accounting.
Zuzu owes me 2 minas 13 1/3 shekels of
rev.
silver.
1 mina of silver is owed by
amas-bàni son of Puzur-Ištar.
20
1/2 minas 5 shekels of silver is
owed by Illil-bàni.
The 1 mina of silver which he borrowed
from you,
the 1/2 mina which Aššur-samsi has
brought to you—in tin it was
25
8 minas—plus the 6 minas of tin
that you will claim for the enterprise—
take that, and then add in all
14 minas 8 shekels of tin to the
silver, seal it for the transportation,
30
and send it to me as a greeting
as soon as you arrive.
If you are truly
my brother, do not make me angry.
1.e.
If Samas-bàni is absent, then make
available a corresponding amount
35
of 1 mina of silver and send it to me.
Please, let my share be deposited at the time
of the accounting."""),
]


DERCKSEN_TEXTS = [
    ("Kt c/k 763", """50 zamrutu, 2 5/6 minas of silver, 6 shekels of silver for sheep, all this I gave in Qattara. 1
sila of fine (oil) of ... for Ala-Adda ...; [I gave] 17 zamrutu for the caravan in Našila.
(too broken for translation)
... 6 zamrutu ... in Hazura; all this I gave to the caravan."""),
    ("Kt c/k 399", """x sila of aromatic oil, 5 dulbatum and sundries I gave in Šerwun.
3 shekels of tin I gave to the servant of the kaššum in Inum.
3 bronze nails and 2 g., two-thirds mina of tin to the nākirum, I gave in Bābum.
I paid 5 shekels of tin when I went on a mission.
3 shekels of silver I paid in Burallum for wine after our accounts had been settled."""),
    ("Kt c/k 216", """[...] for wine. 3 shekels of tin, 5 dulbatum and
sundries for Panaka. 5 dulbatum and sundries for Enlil-bāni in Humahum. One-third of a mina of
copper for where our donkeys were standing in Humahum. 15 shekels of copper in Zurzu. Half a
mina of copper to the massu’um of Ašihum. Half a mina minus half a shekel of tin, 10 dulbatum
and 2 containers(?) of first quality oil and sundries for the palace, 5 shekels of tin, 5 dulbatum and
one container(?) of first quality oil and sundries for the kaššum in Hurumhašum."""),
    ("Kt c/k 441", """11 shekels of tin to the servant of Sam’al; 5 shekels of tin and a seal of haematite for his
compagnon; 1 mina 2 shekels of tin in Enumgalim for ummanum. 5 beads of carnelian and 2 of
zikanšarri for wood in Šunukam. 5 dulbatum, 1 muluhum and sundries, a quarter sila of aromatic
oil, 6 ½ shekels of tin for the kaššum of Šunukam. 2 ¼ shekels of tin for wine. 1 ½ shekels of tin
for the wife of the kaššum. 6 ½ shekels of tin for the massu’um in the mountain. 7 dulbatum, 1
muluhum and sundries, 5 shekels of tin to the wife of the brother of the king of Šehwa. 5/6 mina of
tin to the palace of Šehwa; 2 shekels of tin to the great man; 3 shekels of tin to the smith; 3 shekels
of tin for wine; 15 shekels of tin for the kaššum of Šehwa; half a mina and 5 shekels of tin for our
guide – all this I gave upon <our> entering (the town).
5 dulbatum, 2 nigirašum and sundries, 1/3 mina of copper to the elders, 15 shekels of copper
to ...; all this in Tadhul. 3 shekels of tin for wine in Šuhru. 1 sila of aromatic oil, 10 dulbatum, an
amahum, 1 muluhum, sundries and children’s <object> to the palace. 4 G. for the escort, 4
dulbatum and sundries, 2 shekels of tin for the priest. 3 1/3 shekels of tin for wine, 1 2/3 shekels of
tin, also for wine; all this I gave in Burhum."""),
    ("Kt c/k 470+767", """1 shekel of tin to Ilulāya. Half a shekel of tin is with Iliya. I gave him also half a mina of
copper and sundries in Hirašta and one-third mina (of copper?) in Uršu. 10 shekels of tin and 10
dulbatum and sundries to the kaššum of Uršu. 1 ½ shekels of silver export-tax. 1 shekel of silver to
the massu’um. 1 shekel of silver for a flask for the god(?). 2 shekels of tin to Sama. One-third mina
of copper to Inib-Aššur. One-third mina of copper to Gallābum. 1 mina 10 shekels of copper in
Bulbulhum. 5 dulbatum, 1 šinuntum and sundries to the overseer. One-third mina and 3+x shekels
[...] to the massu’um. 1 abarnium-textile, a šitrum, 10 dulbatum to the king. Furthermore, [...] and 1
mina of tin to [...]. A šitrum, 10 dulbatum [...] and sundries to Šuku... . 1 mina 10 shekels of tin, a
šitrum, 10 dulbatum, a muluhum and sundries to the kaššum [...]. 1 šitrum to the kaššum [...]. 3 ½
shekels of tin [...] ... to Madāya. All this I paid in Mama."""),
    ("Kt c/k 766", """12 shekels of copper for the inn. 12 shekels of copper at the bridge. 12 shekels of copper in
Bulbulhum. Half a mina of copper for garments. One-third shekel (of silver) for the inn in Aruwar.
We took 10 shekels of silver for our food. Half a shekel of silver šaddu’atum-tax. Half a shekel of
silver for the inn in Uršu. 2 shekels of silver were lost during refining."""),
]


def log(message: str) -> None:
    print(message, flush=True)
    with OUTPUT_LOG.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def normalize_id(text: str) -> str:
    text = str(text).lower().strip()
    text = text.split("|")[0].strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text


def clean_translation(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n(?:lo\.e\.|rev\.|u\.e\.|l\.e\.|1\.e\.)", "\n", text, flags=re.I)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n\s*\d+\s+", " ", text)
    text = re.sub(r"-\n", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_published_texts() -> list[dict[str, str]]:
    rows = []
    with (RAW_DATA_DIR / "published_texts.csv").open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def load_train_oare_ids() -> set[str]:
    ids = set()
    with (RAW_DATA_DIR / "train.csv").open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ids.add(row["oare_id"])
    return ids


def build_text_index(published_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    alias_index = {}
    excavation_index = {}
    for row in published_rows:
        aliases = row.get("aliases", "") or ""
        for alias in re.split(r"[|;]", aliases):
            alias_norm = normalize_id(alias)
            if alias_norm:
                alias_index[alias_norm] = row
        excavation = (row.get("excavation_no", "") or "").strip()
        if excavation:
            excavation_index[excavation] = row
    return alias_index, excavation_index


def build_try5_pairs() -> list[dict[str, str]]:
    published_rows = load_published_texts()
    train_ids = load_train_oare_ids()
    alias_index, excavation_index = build_text_index(published_rows)
    pairs = []

    for text_id, translation in LARSEN_TEXTS:
        match = alias_index.get(normalize_id(text_id))
        if not match:
            continue
        pairs.append(
            {
                "source": "archive_alias",
                "match_key": text_id,
                "oare_id": match["oare_id"],
                "is_new_vs_train": str(match["oare_id"] not in train_ids).lower(),
                "transliteration": match["transliteration"],
                "translation": clean_translation(translation),
                "aliases": match.get("aliases", ""),
                "excavation_no": match.get("excavation_no", ""),
            }
        )

    for excavation_no, translation in DERCKSEN_TEXTS:
        match = excavation_index.get(excavation_no)
        if not match:
            continue
        pairs.append(
            {
                "source": "excavation_match",
                "match_key": excavation_no,
                "oare_id": match["oare_id"],
                "is_new_vs_train": str(match["oare_id"] not in train_ids).lower(),
                "transliteration": match["transliteration"],
                "translation": clean_translation(translation),
                "aliases": match.get("aliases", ""),
                "excavation_no": match.get("excavation_no", ""),
            }
        )

    deduped = []
    seen = set()
    for row in pairs:
        if row["oare_id"] in seen:
            continue
        seen.add(row["oare_id"])
        deduped.append(row)
    return deduped


def write_outputs(rows: list[dict[str, str]], elapsed: float) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source",
                "match_key",
                "oare_id",
                "is_new_vs_train",
                "aliases",
                "excavation_no",
                "transliteration",
                "translation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    alias_count = sum(row["source"] == "archive_alias" for row in rows)
    excavation_count = sum(row["source"] == "excavation_match" for row in rows)
    new_count = sum(row["is_new_vs_train"] == "true" for row in rows)

    with OUTPUT_READ.open("w", encoding="utf-8") as file:
        file.write("# Try 5 Archive Expansion Log\n\n")
        file.write("## Scope\n")
        file.write("- Attempt: Try 5\n")
        file.write("- Goal: build a supplemental parallel corpus using metadata-driven archive matching rather than blind OCR line mining\n")
        file.write("- Strategy: pair trusted transliterations from `published_texts.csv` with publication-derived translations through aliases and excavation identifiers\n\n")

        file.write("## Techniques Used\n")
        file.write("- Worked archive-by-archive instead of scanning the full OCR dump indiscriminately\n")
        file.write("- Matched clean transliterations through metadata keys rather than fuzzy transliteration-only alignment\n")
        file.write("- Normalized aliases before exact matching\n")
        file.write("- Used excavation numbers as a second stable identifier source\n")
        file.write("- Cleaned the translation side by removing line numbers, running headers like `lo.e.`/`rev.`/`u.e.`/`l.e.`, and broken line wraps\n")
        file.write("- Kept the resulting corpus separate as a supplemental archive-level dataset\n\n")

        file.write("## Results\n")
        file.write(f"- Total matched supplemental texts: {len(rows)}\n")
        file.write(f"- Alias-driven archive matches: {alias_count}\n")
        file.write(f"- Excavation-number matches: {excavation_count}\n")
        file.write(f"- New records relative to `train.csv`: {new_count}\n")
        file.write(f"- Runtime: {elapsed:.2f} seconds\n\n")

        file.write("## Sample Records\n")
        for row in rows[:5]:
            file.write(f"- KEY: {row['match_key']}\n")
            file.write(f"  oare_id={row['oare_id']}, new_vs_train={row['is_new_vs_train']}\n")
            file.write(f"  transliteration={row['transliteration'][:220]}\n")
            file.write(f"  translation={row['translation'][:220]}\n")


def main() -> None:
    start = time.time()
    OUTPUT_LOG.write_text("", encoding="utf-8")

    log("TRY 5: Building metadata-driven archive expansion set")
    rows = build_try5_pairs()
    elapsed = time.time() - start
    write_outputs(rows, elapsed)
    log(f"TRY 5 COMPLETE: matched supplemental texts={len(rows)} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
