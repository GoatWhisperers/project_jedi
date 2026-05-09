"""
probe_minimal_pairs_gemma4.py

Estrae vettori concettuali da Gemma4-E4B-IT usando coppie minimali:
stessa frase, solo la parola chiave cambia. Molto più pulito del dataset
500+500 generico perché elimina il rumore contestuale.

Output: output/vector_library_minimal/{category}/{concept}/gemma4-e4b-it/
"""

import numpy as np
import requests
import os
import json
import time

MANAGER_URL = "http://localhost:8020"
LAYERS = list(range(29, 42))   # L29-L41
TOKEN_POS = "mean"
OUT_ROOT = "output/vector_library_minimal"

# ── Dataset coppie minimali ────────────────────────────────────────────────────

CONCEPTS = {

"hot_vs_cold": {
  "category": "sensoriale",
  "pairs": [
    ("The water is hot.",              "The water is cold."),
    ("I feel hot right now.",          "I feel cold right now."),
    ("The coffee is hot.",             "The coffee is cold."),
    ("Her hands were hot.",            "Her hands were cold."),
    ("The metal felt hot.",            "The metal felt cold."),
    ("The air is hot today.",          "The air is cold today."),
    ("He touched something hot.",      "He touched something cold."),
    ("The soup is hot.",               "The soup is cold."),
    ("The sand was hot.",              "The sand was cold."),
    ("The engine runs hot.",           "The engine runs cold."),
    ("The pavement felt hot.",         "The pavement felt cold."),
    ("Her skin was hot.",              "Her skin was cold."),
    ("The stone is hot.",              "The stone is cold."),
    ("The wind felt hot.",             "The wind felt cold."),
    ("The ground is hot.",             "The ground is cold."),
    ("The pipe is hot.",               "The pipe is cold."),
    ("The surface was hot.",           "The surface was cold."),
    ("His forehead is hot.",           "His forehead is cold."),
    ("The plate is hot.",              "The plate is cold."),
    ("The room feels hot.",            "The room feels cold."),
    ("The spring water is hot.",       "The spring water is cold."),
    ("The floor was hot.",             "The floor was cold."),
    ("The breath was hot.",            "The breath was cold."),
    ("The rod was hot.",               "The rod was cold."),
    ("The blanket felt hot.",          "The blanket felt cold."),
    ("The glass is hot.",              "The glass is cold."),
    ("The car hood is hot.",           "The car hood is cold."),
    ("The shower is hot.",             "The shower is cold."),
    ("The seat was hot.",              "The seat was cold."),
    ("The handle felt hot.",           "The handle felt cold."),
    ("The tea is hot.",                "The tea is cold."),
    ("The radiator is hot.",           "The radiator is cold."),
    ("The air coming out is hot.",     "The air coming out is cold."),
    ("The towel was hot.",             "The towel was cold."),
    ("The rock surface is hot.",       "The rock surface is cold."),
    ("The tap water is hot.",          "The tap water is cold."),
    ("The exhaust is hot.",            "The exhaust is cold."),
    ("The iron is hot.",               "The iron is cold."),
    ("The bath water is hot.",         "The bath water is cold."),
    ("The steel beam is hot.",         "The steel beam is cold."),
    ("The liquid inside is hot.",      "The liquid inside is cold."),
    ("The coil is hot.",               "The coil is cold."),
    ("The asphalt is hot.",            "The asphalt is cold."),
    ("The vent blows hot air.",        "The vent blows cold air."),
    ("The spring felt hot.",           "The spring felt cold."),
    ("The wire is hot.",               "The wire is cold."),
    ("The sand underfoot is hot.",     "The sand underfoot is cold."),
    ("The surface of the pan is hot.", "The surface of the pan is cold."),
    ("The ember was hot.",             "The ember was cold."),
    ("The jet of steam is hot.",       "The jet of steam is cold."),
  ]
},

"luce_vs_buio": {
  "category": "sensoriale",
  "pairs": [
    ("The room is bright.",            "The room is dark."),
    ("The sky is bright today.",       "The sky is dark today."),
    ("The screen is bright.",          "The screen is dark."),
    ("Her eyes were bright.",          "Her eyes were dark."),
    ("The lamp was bright.",           "The lamp was dark."),
    ("The corridor is bright.",        "The corridor is dark."),
    ("The surface looked bright.",     "The surface looked dark."),
    ("The window lets in bright light.","The window lets in no light."),
    ("The path ahead is bright.",      "The path ahead is dark."),
    ("The reflection was bright.",     "The reflection was dark."),
    ("The hall was bright.",           "The hall was dark."),
    ("The kitchen is bright.",         "The kitchen is dark."),
    ("The stage is bright.",           "The stage is dark."),
    ("The field was bright.",          "The field was dark."),
    ("The object appeared bright.",    "The object appeared dark."),
    ("The ceiling felt bright.",       "The ceiling felt dark."),
    ("The alley was bright.",          "The alley was dark."),
    ("The wall is bright.",            "The wall is dark."),
    ("The space felt bright.",         "The space felt dark."),
    ("The interior was bright.",       "The interior was dark."),
    ("The area is bright.",            "The area is dark."),
    ("The spot is bright.",            "The spot is dark."),
    ("The outline was bright.",        "The outline was dark."),
    ("The entrance is bright.",        "The entrance is dark."),
    ("The background is bright.",      "The background is dark."),
    ("The tunnel was bright.",         "The tunnel was dark."),
    ("The chamber is bright.",         "The chamber is dark."),
    ("The cellar was bright.",         "The cellar was dark."),
    ("The sky at noon is bright.",     "The sky at midnight is dark."),
    ("The forest clearing is bright.", "The forest clearing is dark."),
    ("The courtyard is bright.",       "The courtyard is dark."),
    ("The shed was bright.",           "The shed was dark."),
    ("The office is bright.",          "The office is dark."),
    ("The stairwell is bright.",       "The stairwell is dark."),
    ("The attic is bright.",           "The attic is dark."),
    ("The garage is bright.",          "The garage is dark."),
    ("The basement is bright.",        "The basement is dark."),
    ("The studio is bright.",          "The studio is dark."),
    ("The theater is bright.",         "The theater is dark."),
    ("The warehouse is bright.",       "The warehouse is dark."),
    ("The classroom is bright.",       "The classroom is dark."),
    ("The hallway is bright.",         "The hallway is dark."),
    ("The bathroom is bright.",        "The bathroom is dark."),
    ("The bedroom is bright.",         "The bedroom is dark."),
    ("The living room is bright.",     "The living room is dark."),
    ("The closet is bright.",          "The closet is dark."),
    ("The passage was bright.",        "The passage was dark."),
    ("The bay is bright.",             "The bay is dark."),
    ("The cave was bright.",           "The cave was dark."),
    ("The valley is bright.",          "The valley is dark."),
  ]
},

"calma_vs_allerta": {
  "category": "sensoriale",
  "pairs": [
    ("She felt calm.",                 "She felt alert."),
    ("The atmosphere was calm.",       "The atmosphere was tense."),
    ("He was calm.",                   "He was alarmed."),
    ("Her breathing was calm.",        "Her breathing was rapid."),
    ("The city was calm.",             "The city was on alert."),
    ("The crowd was calm.",            "The crowd was agitated."),
    ("The child was calm.",            "The child was startled."),
    ("The dog was calm.",              "The dog was alert."),
    ("His tone was calm.",             "His tone was urgent."),
    ("The patient was calm.",          "The patient was anxious."),
    ("The meeting was calm.",          "The meeting was tense."),
    ("The water was calm.",            "The water was turbulent."),
    ("Her mind was calm.",             "Her mind was racing."),
    ("The soldier was calm.",          "The soldier was on guard."),
    ("The driver was calm.",           "The driver was alarmed."),
    ("The night was calm.",            "The night was uneasy."),
    ("His pulse was calm.",            "His pulse was racing."),
    ("The village was calm.",          "The village was on alert."),
    ("The horse was calm.",            "The horse was startled."),
    ("The pilot was calm.",            "The pilot was alarmed."),
    ("The air was calm.",              "The air was charged."),
    ("The student was calm.",          "The student was anxious."),
    ("The market was calm.",           "The market was chaotic."),
    ("Her face was calm.",             "Her face was tense."),
    ("The sea was calm.",              "The sea was rough."),
    ("The house was calm.",            "The house was tense."),
    ("The surgeon was calm.",          "The surgeon was alert."),
    ("The guard was calm.",            "The guard was alert."),
    ("The forest was calm.",           "The forest was restless."),
    ("The mood was calm.",             "The mood was frantic."),
    ("The reaction was calm.",         "The reaction was sharp."),
    ("The response was calm.",         "The response was urgent."),
    ("The team was calm.",             "The team was on edge."),
    ("The room was calm.",             "The room was tense."),
    ("The animal was calm.",           "The animal was agitated."),
    ("His posture was calm.",          "His posture was rigid."),
    ("The street was calm.",           "The street was frantic."),
    ("The environment was calm.",      "The environment was stressful."),
    ("The speaker was calm.",          "The speaker was alarmed."),
    ("The leader was calm.",           "The leader was anxious."),
    ("The officer was calm.",          "The officer was alert."),
    ("The situation was calm.",        "The situation was urgent."),
    ("The clinic was calm.",           "The clinic was busy."),
    ("The teacher was calm.",          "The teacher was frantic."),
    ("The cat was calm.",              "The cat was alert."),
    ("The baby was calm.",             "The baby was startled."),
    ("The lake was calm.",             "The lake was choppy."),
    ("The evening was calm.",          "The evening was tense."),
    ("The border was calm.",           "The border was on alert."),
    ("The system was calm.",           "The system was alarmed."),
  ]
},

"liscio_vs_ruvido": {
  "category": "sensoriale",
  "pairs": [
    ("The surface is smooth.",         "The surface is rough."),
    ("The stone felt smooth.",         "The stone felt rough."),
    ("The fabric is smooth.",          "The fabric is rough."),
    ("The road is smooth.",            "The road is rough."),
    ("The wall feels smooth.",         "The wall feels rough."),
    ("The wood is smooth.",            "The wood is rough."),
    ("The glass is smooth.",           "The glass is rough."),
    ("The skin was smooth.",           "The skin was rough."),
    ("The paper is smooth.",           "The paper is rough."),
    ("The plastic is smooth.",         "The plastic is rough."),
    ("The metal surface is smooth.",   "The metal surface is rough."),
    ("The bark felt smooth.",          "The bark felt rough."),
    ("The floor is smooth.",           "The floor is rough."),
    ("The table is smooth.",           "The table is rough."),
    ("The tile is smooth.",            "The tile is rough."),
    ("The edge is smooth.",            "The edge is rough."),
    ("The finish is smooth.",          "The finish is rough."),
    ("The terrain is smooth.",         "The terrain is rough."),
    ("The ice is smooth.",             "The ice is rough."),
    ("The leather is smooth.",         "The leather is rough."),
    ("The pebble was smooth.",         "The pebble was rough."),
    ("The coating is smooth.",         "The coating is rough."),
    ("The concrete is smooth.",        "The concrete is rough."),
    ("The paint feels smooth.",        "The paint feels rough."),
    ("The shell was smooth.",          "The shell was rough."),
    ("The plank is smooth.",           "The plank is rough."),
    ("The grip is smooth.",            "The grip is rough."),
    ("The touchpad is smooth.",        "The touchpad is rough."),
    ("The curve is smooth.",           "The curve is rough."),
    ("The coating felt smooth.",       "The coating felt rough."),
    ("The clay was smooth.",           "The clay was rough."),
    ("The wax made it smooth.",        "The sandpaper made it rough."),
    ("The handle is smooth.",          "The handle is rough."),
    ("The silk is smooth.",            "The silk is rough."),
    ("The rubber is smooth.",          "The rubber is rough."),
    ("The sand was smooth.",           "The sand was coarse."),
    ("The rock was smooth.",           "The rock was jagged."),
    ("The bolt is smooth.",            "The bolt is rough."),
    ("The rim is smooth.",             "The rim is rough."),
    ("The joint felt smooth.",         "The joint felt rough."),
    ("The path is smooth.",            "The path is rough."),
    ("The clay felt smooth.",          "The clay felt gritty."),
    ("The cap is smooth.",             "The cap is rough."),
    ("The board is smooth.",           "The board is rough."),
    ("The cylinder is smooth.",        "The cylinder is rough."),
    ("The surface after polishing is smooth.", "The surface before polishing is rough."),
    ("The chip is smooth.",            "The chip is rough."),
    ("The blade is smooth.",           "The blade is rough."),
    ("The bar is smooth.",             "The bar is rough."),
    ("The sphere is smooth.",          "The sphere is rough."),
  ]
},

"secco_vs_umido": {
  "category": "sensoriale",
  "pairs": [
    ("The air is dry.",                "The air is humid."),
    ("The towel is dry.",              "The towel is wet."),
    ("The ground is dry.",             "The ground is wet."),
    ("The skin feels dry.",            "The skin feels moist."),
    ("The wood is dry.",               "The wood is damp."),
    ("The desert is dry.",             "The jungle is humid."),
    ("The cloth is dry.",              "The cloth is damp."),
    ("The sand is dry.",               "The sand is wet."),
    ("The soil is dry.",               "The soil is moist."),
    ("The grass is dry.",              "The grass is wet."),
    ("The paper is dry.",              "The paper is damp."),
    ("The bread is dry.",              "The bread is moist."),
    ("The climate is dry.",            "The climate is humid."),
    ("The road is dry.",               "The road is wet."),
    ("The hay is dry.",                "The hay is damp."),
    ("The concrete is dry.",           "The concrete is wet."),
    ("The sponge is dry.",             "The sponge is wet."),
    ("The mat is dry.",                "The mat is wet."),
    ("The rock is dry.",               "The rock is wet."),
    ("The wall is dry.",               "The wall is damp."),
    ("The leaf is dry.",               "The leaf is wet."),
    ("The rope is dry.",               "The rope is wet."),
    ("The eye is dry.",                "The eye is moist."),
    ("The wind is dry.",               "The wind is humid."),
    ("The season is dry.",             "The season is wet."),
    ("The summer air is dry.",         "The summer air is humid."),
    ("The cabin air is dry.",          "The cabin air is humid."),
    ("The throat is dry.",             "The throat is moist."),
    ("The paint is dry.",              "The paint is wet."),
    ("The mud is dry.",                "The mud is wet."),
    ("The cellar is dry.",             "The cellar is damp."),
    ("The crack in the soil is dry.",  "The crack in the soil is wet."),
    ("The sack is dry.",               "The sack is wet."),
    ("The corn is dry.",               "The corn is moist."),
    ("The basement is dry.",           "The basement is damp."),
    ("The bark is dry.",               "The bark is wet."),
    ("The air inside is dry.",         "The air inside is humid."),
    ("The fabric feels dry.",          "The fabric feels damp."),
    ("The stone is dry.",              "The stone is wet."),
    ("The loam is dry.",               "The loam is moist."),
    ("The cave is dry.",               "The cave is damp."),
    ("The tongue feels dry.",          "The tongue feels moist."),
    ("The hay bale is dry.",           "The hay bale is damp."),
    ("The grain is dry.",              "The grain is wet."),
    ("The earth is dry.",              "The earth is moist."),
    ("The peat is dry.",               "The peat is wet."),
    ("The forest floor is dry.",       "The forest floor is wet."),
    ("The fur is dry.",                "The fur is wet."),
    ("The moss is dry.",               "The moss is wet."),
    ("The mattress is dry.",           "The mattress is damp."),
  ]
},

"duro_vs_morbido": {
  "category": "sensoriale",
  "pairs": [
    ("The material is hard.",          "The material is soft."),
    ("The stone is hard.",             "The stone is soft."),
    ("The bread is hard.",             "The bread is soft."),
    ("The mattress is hard.",          "The mattress is soft."),
    ("The ground is hard.",            "The ground is soft."),
    ("The shell is hard.",             "The shell is soft."),
    ("The pillow is hard.",            "The pillow is soft."),
    ("The surface feels hard.",        "The surface feels soft."),
    ("The wood is hard.",              "The wood is soft."),
    ("The rubber is hard.",            "The rubber is soft."),
    ("The clay is hard.",              "The clay is soft."),
    ("The cheese is hard.",            "The cheese is soft."),
    ("The cushion is hard.",           "The cushion is soft."),
    ("The wall is hard.",              "The wall is soft."),
    ("The metal is hard.",             "The metal is soft."),
    ("The seat is hard.",              "The seat is soft."),
    ("The floor is hard.",             "The floor is soft."),
    ("The bed is hard.",               "The bed is soft."),
    ("The soil is hard.",              "The soil is soft."),
    ("The wax is hard.",               "The wax is soft."),
    ("The grip felt hard.",            "The grip felt soft."),
    ("The tissue is hard.",            "The tissue is soft."),
    ("The crust is hard.",             "The crust is soft."),
    ("The foam is hard.",              "The foam is soft."),
    ("The object felt hard.",          "The object felt soft."),
    ("The fruit is hard.",             "The fruit is soft."),
    ("The ball is hard.",              "The ball is soft."),
    ("The rock is hard.",              "The rock is soft."),
    ("The table is hard.",             "The table is soft."),
    ("The bone is hard.",              "The bone is soft."),
    ("The bark is hard.",              "The bark is soft."),
    ("The coating is hard.",           "The coating is soft."),
    ("The concrete is hard.",          "The concrete is soft."),
    ("The plastic is hard.",           "The plastic is soft."),
    ("The bread crust is hard.",       "The bread crust is soft."),
    ("The dough is hard.",             "The dough is soft."),
    ("The pencil felt hard.",          "The pencil felt soft."),
    ("The handle is hard.",            "The handle is soft."),
    ("The armor is hard.",             "The armor is soft."),
    ("The shell casing is hard.",      "The shell casing is soft."),
    ("The block is hard.",             "The block is soft."),
    ("The cartilage is hard.",         "The cartilage is soft."),
    ("The lens is hard.",              "The lens is soft."),
    ("The edge is hard.",              "The edge is soft."),
    ("The layer is hard.",             "The layer is soft."),
    ("The nail felt hard.",            "The nail felt soft."),
    ("The biscuit is hard.",           "The biscuit is soft."),
    ("The pellet is hard.",            "The pellet is soft."),
    ("The compound is hard.",          "The compound is soft."),
    ("The surface of the road is hard.","The surface of the road is soft."),
  ]
},

"rumore_vs_silenzio": {
  "category": "uditivo",
  "pairs": [
    ("The room was noisy.",            "The room was silent."),
    ("The street is loud.",            "The street is quiet."),
    ("The crowd is noisy.",            "The crowd is silent."),
    ("The engine is loud.",            "The engine is quiet."),
    ("The hall was noisy.",            "The hall was silent."),
    ("The neighbor is loud.",          "The neighbor is quiet."),
    ("The city is noisy.",             "The city is silent."),
    ("The office is loud.",            "The office is quiet."),
    ("The traffic is noisy.",          "The traffic is silent."),
    ("The party was loud.",            "The party was silent."),
    ("The speaker was loud.",          "The speaker was quiet."),
    ("The machine is noisy.",          "The machine is silent."),
    ("The kitchen was loud.",          "The kitchen was silent."),
    ("The classroom is noisy.",        "The classroom is silent."),
    ("The playground is loud.",        "The playground is quiet."),
    ("The market was noisy.",          "The market was silent."),
    ("The workshop is loud.",          "The workshop is quiet."),
    ("The bar was noisy.",             "The bar was silent."),
    ("The station is loud.",           "The station is quiet."),
    ("The airport was noisy.",         "The airport was silent."),
    ("The factory is loud.",           "The factory is quiet."),
    ("The gym is noisy.",              "The gym is silent."),
    ("The school was loud.",           "The school was silent."),
    ("The restaurant was noisy.",      "The restaurant was silent."),
    ("The stadium is loud.",           "The stadium is quiet."),
    ("The corridor is noisy.",         "The corridor is silent."),
    ("The bus was loud.",              "The bus was quiet."),
    ("The cafeteria is noisy.",        "The cafeteria is silent."),
    ("The library was loud.",          "The library was silent."),
    ("The concert was noisy.",         "The concert was silent."),
    ("The alley was noisy.",           "The alley was silent."),
    ("The construction is loud.",      "The construction is quiet."),
    ("The truck is noisy.",            "The truck is quiet."),
    ("The drill is loud.",             "The drill is quiet."),
    ("The bird was loud.",             "The bird was silent."),
    ("The dog was noisy.",             "The dog was silent."),
    ("The car was loud.",              "The car was quiet."),
    ("The fan is loud.",               "The fan is quiet."),
    ("The dishwasher is noisy.",       "The dishwasher is silent."),
    ("The AC unit is loud.",           "The AC unit is quiet."),
    ("The phone was loud.",            "The phone was silent."),
    ("The alarm was loud.",            "The alarm was silent."),
    ("The TV was noisy.",              "The TV was silent."),
    ("The kettle is loud.",            "The kettle is quiet."),
    ("The pipes are noisy.",           "The pipes are silent."),
    ("The wind was loud.",             "The wind was silent."),
    ("The rain was noisy.",            "The rain was silent."),
    ("The crowd outside is loud.",     "The crowd outside is quiet."),
    ("The child was noisy.",           "The child was silent."),
    ("The hall outside was loud.",     "The hall outside was silent."),
  ]
},

"dolce_vs_amaro": {
  "category": "gustativo",
  "pairs": [
    ("The taste is sweet.",            "The taste is bitter."),
    ("The coffee is sweet.",           "The coffee is bitter."),
    ("The juice is sweet.",            "The juice is bitter."),
    ("The fruit is sweet.",            "The fruit is bitter."),
    ("The chocolate is sweet.",        "The chocolate is bitter."),
    ("The syrup is sweet.",            "The syrup is bitter."),
    ("The drink is sweet.",            "The drink is bitter."),
    ("The candy is sweet.",            "The candy is bitter."),
    ("The flavor is sweet.",           "The flavor is bitter."),
    ("The wine is sweet.",             "The wine is bitter."),
    ("The herb is sweet.",             "The herb is bitter."),
    ("The tea is sweet.",              "The tea is bitter."),
    ("The berry is sweet.",            "The berry is bitter."),
    ("The sauce is sweet.",            "The sauce is bitter."),
    ("The liquid tasted sweet.",       "The liquid tasted bitter."),
    ("The medicine was sweet.",        "The medicine was bitter."),
    ("The residue was sweet.",         "The residue was bitter."),
    ("The root is sweet.",             "The root is bitter."),
    ("The coating is sweet.",          "The coating is bitter."),
    ("The cream is sweet.",            "The cream is bitter."),
    ("The paste is sweet.",            "The paste is bitter."),
    ("The aftertaste is sweet.",       "The aftertaste is bitter."),
    ("The powder is sweet.",           "The powder is bitter."),
    ("The compound is sweet.",         "The compound is bitter."),
    ("The extract is sweet.",          "The extract is bitter."),
    ("The mix is sweet.",              "The mix is bitter."),
    ("The seed is sweet.",             "The seed is bitter."),
    ("The water tasted sweet.",        "The water tasted bitter."),
    ("The liquid feels sweet.",        "The liquid feels bitter."),
    ("The rinse was sweet.",           "The rinse was bitter."),
    ("The rind is sweet.",             "The rind is bitter."),
    ("The gel is sweet.",              "The gel is bitter."),
    ("The compound tasted sweet.",     "The compound tasted bitter."),
    ("The pill was sweet.",            "The pill was bitter."),
    ("The drop was sweet.",            "The drop was bitter."),
    ("The sip was sweet.",             "The sip was bitter."),
    ("The taste of the leaf is sweet.","The taste of the leaf is bitter."),
    ("The nectar is sweet.",           "The nectar is bitter."),
    ("The juice of the fruit is sweet.","The juice of the fruit is bitter."),
    ("The coating on the pill is sweet.","The coating on the pill is bitter."),
    ("The aftertaste of the drink is sweet.","The aftertaste of the drink is bitter."),
    ("The finish of the wine is sweet.","The finish of the wine is bitter."),
    ("The note in the flavor is sweet.","The note in the flavor is bitter."),
    ("The undertone is sweet.",        "The undertone is bitter."),
    ("The edge of the taste is sweet.","The edge of the taste is bitter."),
    ("The first impression is sweet.", "The first impression is bitter."),
    ("The lingering taste is sweet.",  "The lingering taste is bitter."),
    ("The base flavor is sweet.",      "The base flavor is bitter."),
    ("The essence is sweet.",          "The essence is bitter."),
    ("The component is sweet.",        "The component is bitter."),
  ]
},

"odore_forte_vs_inodore": {
  "category": "olfattivo",
  "pairs": [
    ("The smell is strong.",           "There is no smell."),
    ("The odor is strong.",            "There is no odor."),
    ("The scent is strong.",           "There is no scent."),
    ("The fragrance is strong.",       "There is no fragrance."),
    ("The aroma is intense.",          "There is no aroma."),
    ("The room smells strongly.",      "The room has no smell."),
    ("The substance has a strong smell.","The substance has no smell."),
    ("The chemical has a strong odor.","The chemical has no odor."),
    ("The flower has a strong scent.", "The flower has no scent."),
    ("The food has a strong smell.",   "The food has no smell."),
    ("The perfume is strong.",         "The perfume is absent."),
    ("The gas has a strong odor.",     "The gas has no odor."),
    ("The smoke has a strong smell.",  "The smoke has no smell."),
    ("The herb has a strong aroma.",   "The herb has no aroma."),
    ("The sweat has a strong odor.",   "The sweat has no odor."),
    ("The paint has a strong smell.",  "The paint has no smell."),
    ("The soil has a strong smell.",   "The soil has no smell."),
    ("The cheese has a strong odor.",  "The cheese has no odor."),
    ("The fish has a strong smell.",   "The fish has no smell."),
    ("The wood has a strong aroma.",   "The wood has no aroma."),
    ("The wine has a strong bouquet.", "The wine has no bouquet."),
    ("The flower releases a strong scent.","The flower releases no scent."),
    ("The liquid has a strong odor.",  "The liquid has no odor."),
    ("The compound has a strong smell.","The compound has no smell."),
    ("The air is filled with a strong scent.","The air has no scent."),
    ("The candle has a strong fragrance.","The candle has no fragrance."),
    ("The spice has a strong aroma.",  "The spice has no aroma."),
    ("The cleaning agent has a strong odor.","The cleaning agent has no odor."),
    ("The meat has a strong smell.",   "The meat has no smell."),
    ("The exhaust has a strong odor.", "The exhaust has no odor."),
    ("The rubber has a strong smell.", "The rubber has no smell."),
    ("The oil has a strong odor.",     "The oil has no odor."),
    ("The garlic has a strong smell.", "The garlic has no smell."),
    ("The onion has a strong odor.",   "The onion has no odor."),
    ("The vinegar has a strong smell.","The vinegar has no smell."),
    ("The ammonia has a strong odor.", "The ammonia has no odor."),
    ("The turpentine has a strong smell.","The turpentine has no smell."),
    ("The leather has a strong scent.","The leather has no scent."),
    ("The mold has a strong odor.",    "The mold has no odor."),
    ("The medicine has a strong smell.","The medicine has no smell."),
    ("The bleach has a strong odor.",  "The bleach has no odor."),
    ("The trash has a strong smell.",  "The trash has no smell."),
    ("The sewage has a strong odor.",  "The sewage has no odor."),
    ("The flower bed has a strong fragrance.","The flower bed has no fragrance."),
    ("The bakery has a strong aroma.", "The bakery has no aroma."),
    ("The forest has a strong scent.", "The forest has no scent."),
    ("The damp room has a strong odor.","The dry room has no odor."),
    ("The shampoo has a strong fragrance.","The shampoo has no fragrance."),
    ("The detergent has a strong smell.","The detergent has no smell."),
    ("The essential oil has a strong aroma.","The essential oil has no aroma."),
  ]
},

}  # fine CONCEPTS


# ── Funzioni ───────────────────────────────────────────────────────────────────

def check_gpu_free():
    r = requests.get(f"{MANAGER_URL}/api/status", timeout=10)
    d = r.json()
    if d.get("busy"):
        raise RuntimeError(f"GPU busy: {d.get('busy_owner')}")
    if d.get("model") != "Gemma4-E4B-IT":
        raise RuntimeError(f"Modello sbagliato: {d.get('model')}")
    return True


def extract(pos_sents, neg_sents, layers):
    r = requests.post(f"{MANAGER_URL}/api/extract_activations", json={
        "sentences_pos":  pos_sents,
        "sentences_neg":  neg_sents,
        "layers":         layers,
        "token_position": TOKEN_POS,
    }, timeout=600)
    return r.json()


def compute_stats(pos, neg):
    diffs = pos - neg
    norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8
    unit_diffs = diffs / norms
    n = len(pos)
    cos_mat = unit_diffs @ unit_diffs.T
    np.fill_diagonal(cos_mat, 0)
    coherence = float(cos_mat.sum() / (n * (n - 1)))

    mean_diff = pos.mean(0) - neg.mean(0)
    unit_vec  = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)

    proj_pos = pos @ unit_vec
    proj_neg = neg @ unit_vec
    sep   = float(proj_pos.mean() - proj_neg.mean())
    noise = float((proj_pos.std() + proj_neg.std()) / 2 + 1e-8)
    snr   = sep / noise

    cos_means = float(np.dot(pos.mean(0) / np.linalg.norm(pos.mean(0)),
                             neg.mean(0) / np.linalg.norm(neg.mean(0))))
    return unit_vec, coherence, snr, cos_means


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PROBE MINIMAL PAIRS — Gemma4-E4B-IT ===\n")

    summary = {}

    for concept, info in CONCEPTS.items():
        category = info["category"]
        pairs    = info["pairs"]
        pos_sents = [p[0] for p in pairs]
        neg_sents = [p[1] for p in pairs]

        print(f"\n{'─'*60}")
        print(f"  {concept}  ({len(pairs)} coppie minimali)")
        print(f"{'─'*60}")

        check_gpu_free()

        t0 = time.time()
        data = extract(pos_sents, neg_sents, LAYERS)
        print(f"  Estrazione completata in {time.time()-t0:.0f}s")

        out_dir = os.path.join(OUT_ROOT, category, concept, "gemma4-e4b-it")
        os.makedirs(out_dir, exist_ok=True)

        print(f"  {'Layer':>5}  {'coherence':>9}  {'SNR':>8}  {'cos(µ+,µ-)':>11}")
        print(f"  {'─'*5}  {'─'*9}  {'─'*8}  {'─'*11}")

        best_snr, best_layer = -99, None
        layer_results = {}

        for lid in LAYERS:
            pos = np.array(data["pos"][str(lid)], dtype=np.float32)
            neg = np.array(data["neg"][str(lid)], dtype=np.float32)

            unit_vec, coherence, snr, cos_means = compute_stats(pos, neg)

            print(f"  L{lid:>2}:  coh={coherence:+.4f}  SNR={snr:+.4f}  cos={cos_means:+.5f}")
            np.save(os.path.join(out_dir, f"layer_{lid}.npy"), unit_vec)

            layer_results[lid] = {
                "coherence": round(coherence, 4),
                "snr":       round(snr, 4),
                "cos_means": round(cos_means, 5),
            }

            if snr > best_snr:
                best_snr, best_layer = snr, lid

        print(f"\n  *** BEST: L{best_layer}  SNR={best_snr:.4f} ***")
        summary[concept] = {
            "best_layer": best_layer,
            "best_snr":   round(best_snr, 4),
            "layers":     layer_results,
        }

        json.dump({
            "method": "minimal_pairs",
            "n_pairs": len(pairs),
            "best_layer": best_layer,
            "best_snr": round(best_snr, 4),
            "layers": {str(k): {kk: float(vv) for kk, vv in v.items()}
                       for k, v in layer_results.items()},
        }, open(os.path.join(out_dir, "eval_minimal.json"), "w"), indent=2)

    print(f"\n{'='*60}")
    print("  RIEPILOGO FINALE")
    print(f"{'='*60}")
    print(f"  {'Concept':<35}  {'Best L':>6}  {'SNR':>8}  {'Coh':>6}")
    for concept, res in summary.items():
        bl = res["best_layer"]
        snr = res["best_snr"]
        coh = res["layers"][bl]["coherence"]
        print(f"  {concept:<35}  L{bl:>2}     {snr:>8.4f}  {coh:>6.4f}")

    print(f"\nVettori salvati in: {OUT_ROOT}/")
