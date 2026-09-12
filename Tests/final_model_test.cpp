#include"../Engine/Layers/language_model.h"
#include"../Engine/Tokenizer/bpe_tokenizer.h"

#include<cuda_runtime.h>

#include<iostream>
#include<vector>
#include<string>
#include<stdexcept>
#include<cmath>

//============================================================
//Настройкимодели
//============================================================

constsize_tVOCAB_SIZE=1000;
constsize_tEMBED_DIM=128;
constsize_tBLOCKS=4;
constsize_tHEADS=4;
constsize_tHIDDEN=512;

//============================================================
//Настройкигенерации
//============================================================

constintMAX_NEW_TOKENS=100;
constfloatTEMPERATURE=0.8f;
constfloatTOP_P=0.9f;

//-1=неиспользоватьспециальныйendtoken
constintEND_TOKEN_ID=-1;

//============================================================
//Пути
//============================================================

conststd::stringTOKENIZER_PATH=
"../Models/EnglishTokenizer";

conststd::stringMODEL_PATH=
"../Models/EnglishSmaller_best";

//============================================================
//ПроверкаCUDA
//============================================================

voidCheckCUDA(){
cudaError_terror=
cudaGetLastError();

if(error!=cudaSuccess){
throwstd::runtime_error(
std::string("CUDAerror:")+
cudaGetErrorString(error)
);
}
}

//============================================================
//Одинтестгенерации
//============================================================

voidRunGenerationTest(
LanguageModel&model,
BPETokenizer&tokenizer,
conststd::string&prompt,
inttest_number
){
std::cout
<<"\n========================================\n"
<<"GENERATIONTEST"
<<test_number
<<"\n"
<<"========================================\n\n";

std::cout
<<"Prompt:\n"
<<prompt
<<"\n\n";

//========================================================
//Encode
//========================================================

std::vector<size_t>prompt_tokens=
tokenizer.Encode(prompt);

if(prompt_tokens.empty()){
throwstd::runtime_error(
"Promptproducedzerotokens"
);
}

std::cout
<<"Prompttokens:"
<<prompt_tokens.size()
<<"\n";

//========================================================
//Generate
//========================================================

std::vector<size_t>generated=
model.generate(
prompt_tokens,
MAX_NEW_TOKENS,
TEMPERATURE,
TOP_P,
END_TOKEN_ID
);

CheckCUDA();

std::cout
<<"Generatedtokens:"
<<generated.size()
<<"\n\n";

//========================================================
//Полныйтекст
//========================================================

std::vector<size_t>all_tokens;
all_tokens.reserve(
prompt_tokens.size()+
generated.size()
);

all_tokens.insert(
all_tokens.end(),
prompt_tokens.begin(),
prompt_tokens.end()
);

all_tokens.insert(
all_tokens.end(),
generated.begin(),
generated.end()
);

std::stringresult=
tokenizer.Decode(all_tokens);

std::cout
<<"Generatedtext:\n"
<<"----------------------------------------\n"
<<result
<<"\n"
<<"----------------------------------------\n\n";

if(generated.empty()){
throwstd::runtime_error(
"Generationreturnedzerotokens"
);
}

std::cout
<<"[OK]Generationtest"
<<test_number
<<"passed.\n";
}

//============================================================
//main
//============================================================

intmain(){
try{
std::cout
<<"========================================\n"
<<"CUDAGENERATIONTEST\n"
<<"========================================\n\n";

//====================================================
//CUDA
//====================================================

intdevice_count=0;

cudaError_terror=
cudaGetDeviceCount(&device_count);

if(error!=cudaSuccess){
throwstd::runtime_error(
std::string("CUDAerror:")+
cudaGetErrorString(error)
);
}

if(device_count==0){
throwstd::runtime_error(
"NoCUDAdevicesfound"
);
}

cudaDevicePropprop;

error=
cudaGetDeviceProperties(
&prop,
0
);

if(error!=cudaSuccess){
throwstd::runtime_error(
std::string(
"cudaGetDevicePropertiesfailed:"
)+
cudaGetErrorString(error)
);
}

std::cout
<<"GPU:"
<<prop.name
<<"\n\n";

//====================================================
//Tokenizer
//====================================================

BPETokenizertokenizer;

std::cout
<<"Loadingtokenizer...\n";

tokenizer.Load(
TOKENIZER_PATH
);

std::cout
<<"[OK]Tokenizerloaded.\n";

std::cout
<<"Tokenizervocabulary:"
<<tokenizer.GetVocabSize()
<<"\n\n";

if(tokenizer.GetVocabSize()!=VOCAB_SIZE){
throwstd::runtime_error(
"Tokenizervocabularysizedoesnot"
"matchmodelvocabularysize"
);
}

//====================================================
//Model
//====================================================

std::cout
<<"Creatingmodel...\n";

LanguageModelmodel(
VOCAB_SIZE,
EMBED_DIM,
BLOCKS,
HEADS,
HIDDEN,
Device::CUDA
);

std::cout
<<"[OK]Modelcreated.\n\n";

//====================================================
//Loadbestcheckpoint
//====================================================

std::cout
<<"Loadingmodel:\n"
<<MODEL_PATH
<<"\n\n";

model.LoadModel(
MODEL_PATH
);

cudaDeviceSynchronize();

CheckCUDA();

std::cout
<<"[OK]Bestmodelloadedsuccessfully.\n\n";

//====================================================
//Generationsettings
//====================================================

std::cout
<<"========================================\n"
<<"GENERATIONSETTINGS\n"
<<"========================================\n\n";

std::cout
<<"Maxnewtokens:"
<<MAX_NEW_TOKENS
<<"\n";

std::cout
<<"Temperature:"
<<TEMPERATURE
<<"\n";

std::cout
<<"Top-p:"
<<TOP_P
<<"\n";

std::cout
<<"Endtoken:"
<<END_TOKEN_ID
<<"\n\n";

//====================================================
//Prompts
//====================================================

RunGenerationTest(
model,
tokenizer,
"Themanwalkedintotheroomand",
1
);

RunGenerationTest(
model,
tokenizer,
"Idon'tknowwhathappened,but",
2
);

RunGenerationTest(
model,
tokenizer,
"Shelookedathimandsaid",
3
);

RunGenerationTest(
model,
tokenizer,
"Itwasacoldandrainynightwhen",
4
);

//====================================================
//Result
//====================================================

std::cout
<<"\n========================================\n"
<<"RESULT\n"
<<"========================================\n\n";

std::cout
<<"[OK]ALLGENERATIONTESTSPASSED.\n";

std::cout
<<"\n========================================\n"
<<"[OK]GENERATIONTESTPASSED\n"
<<"========================================\n";

return0;
}
catch(conststd::exception&e){
std::cerr
<<"\n========================================\n"
<<"FAILED\n"
<<"========================================\n\n"
<<e.what()
<<"\n";

return1;
}
}