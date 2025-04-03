/* Text Classification using Optimization Methods (AdaGrad, ADAM, etc.)
   Author: Alperen TEKİN
   Date: 2023-11
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#define WORD_LENGTH 30
#define LR 0.05
#define ITER 500
#define EPOCHS 250
#define BATCH_SIZE 35
#define SENTENCE_LENGTH 700

// Type declarations
typedef struct {
    char **wordList;
    int dictSize;
    int wordCounter;
} Dictionary;

typedef struct {
    char **sentenceList;
    int listSize;
    int sentenceCounter;
} SentenceList;

// Function declarations
double dtanh(double x) {
    double coshx = cosh(x);
    return 1.0 / (coshx * coshx);
}

double predictdata(double *trained_W, double *testData, int dictSize);
void gradientDescent(double *W, SentenceList *sentl, Dictionary *dict);
void stochasticGradientDescent(double *W, SentenceList *sentl, Dictionary *dict);
void ADAM(double *W, SentenceList *sentl, Dictionary *dict);
void printHotVector(double *vector, int size);
void createHotVector(Dictionary *dict, char *sentence, double *hotVector);
Dictionary* initializeDictionary();
SentenceList* initializeSentenceList();
bool readDictFile(Dictionary *dict);
bool readSentFile(SentenceList *sentl);
bool expandDict(Dictionary *dict);
bool expandSent(SentenceList *sentl);
bool isExist(char *word, Dictionary *dict);
void printDict(Dictionary *dict);
void trimWord(char *w);

int main() {
    srand(time(NULL));
    int i, totalwordCounter;
    
    Dictionary *dict = initializeDictionary();
    printf("Reading Words for Dictionary from dataset of sentences.\n");
    printf("Reading %s!\n", readDictFile(dict) ? "succeeded" : "failed");
    
    SentenceList *sentl = initializeSentenceList();
    printf("Reading each sentence from text and making a Train sentence list.\n");
    printf("Reading %s!\n", readSentFile(sentl) ? "succeeded" : "failed");
    
    SentenceList *testsentl = initializeSentenceList();
    printf("Reading each sentence from text and making a Test sentence list.\n");
    printf("Reading %s!\n", readSentFile(testsentl) ? "succeeded" : "failed");
    
    double *w = (double*)calloc(dict->wordCounter, sizeof(double));
    printf("Please enter initial w: ");
    scanf("%lf", &w[0]);
    for (i = 1; i < dict->wordCounter; i++) {
        w[i] = w[0];
    }
    
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    
    int choice = 1;
    printf("1-Gradient Descent\n2-Stochastic Gradient Descent\n3-ADAM\nPlease enter the method: ");
    scanf("%d", &choice);
    
    switch(choice) {
        case 1:
            gradientDescent(w, sentl, dict);
            break;
        case 2:
            stochasticGradientDescent(w, sentl, dict);
            break;
        case 3:
            ADAM(w, sentl, dict);
            break;
        default:
            gradientDescent(w, sentl, dict);
    }
    
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("\nExecution time: %f seconds\n", cpu_time_used);
    
    int success = 0;
    double predict;
    char tempSent[SENTENCE_LENGTH];
    for (i = 0; i < testsentl->sentenceCounter; i++) {
        double *testhotvector = (double*)calloc(dict->wordCounter, sizeof(double));
        strcpy(tempSent, testsentl->sentenceList[i]);
        createHotVector(dict, tempSent, testhotvector);
        predict = predictdata(w, testhotvector, dict->wordCounter);
        if (predict < -0.3 && i < 20) {
            success += 1;
        }
        if (predict > 0.3 && i >= 20) {
            success += 1;
        }
        free(testhotvector);
    }
    
    printf("\nYour Success = %d/%d\n", success, testsentl->sentenceCounter);
    
    free(dict);
    free(sentl);
    free(w);
    return 0;
}

bool readDictFile(Dictionary *dict) {
    char _fileName[WORD_LENGTH];
    char buffer[WORD_LENGTH];
    FILE *f_ptr = NULL;
    
    do {
        printf("\n--> Enter the dictionary file name: ");
        scanf("%s", _fileName);
        f_ptr = fopen(_fileName, "r");
        if (f_ptr == NULL)
            printf("\tCouldn't open file, try again\n");
    } while (!f_ptr);
    
    while (fscanf(f_ptr, "%s", buffer) != EOF) {
        trimWord(buffer);
        if (!isExist(buffer, dict) && strlen(buffer) > 0) {
            strcpy(dict->wordList[dict->wordCounter], buffer);
            dict->wordCounter++;
            // Check if more memory is needed
            if (dict->wordCounter + 1 >= dict->dictSize) {
                if (!expandDict(dict)) {
                    fclose(f_ptr);
                    return false;
                }
            }
        }
    }
    fclose(f_ptr);
    return true;
}

bool readSentFile(SentenceList *sentl) {
    char _fileName[WORD_LENGTH];
    FILE *f_ptr = NULL;
    
    do {
        printf("\n--> Enter the sentence file name: ");
        scanf("%s", _fileName);
        f_ptr = fopen(_fileName, "r");
        if (f_ptr == NULL)
            printf("\tCouldn't open file, try again\n");
    } while (!f_ptr);
    
    char sentence[SENTENCE_LENGTH] = {0};
    char ch;
    int i = 0;
    
    while (!feof(f_ptr)) {
        ch = fgetc(f_ptr);
        if (ch != '.' && ch != '?' && ch != '!') {  // add extra punctuation as needed
            if (i == 0 && ch == ' ')
                ; // ignore leading space
            else
                sentence[i++] = ch;
        } else {
            sentence[i] = '\0';
            if (strlen(sentence) > 0) {
                strcpy(sentl->sentenceList[sentl->sentenceCounter], sentence);
                sentl->sentenceCounter++;
                // Check if more memory is needed
                if (sentl->sentenceCounter + 1 >= sentl->listSize) {
                    if (!expandSent(sentl)) {
                        fclose(f_ptr);
                        return false;
                    }
                }
                i = 0;
                sentence[0] = '\0';
            }
        }
    }
    fclose(f_ptr);
    return true;
}

Dictionary* initializeDictionary() {
    Dictionary *dict = (Dictionary*)malloc(sizeof(Dictionary));
    dict->dictSize = 10;
    dict->wordCounter = 0;
    dict->wordList = (char**)malloc(sizeof(char*) * dict->dictSize);
    int i;
    for (i = 0; i < dict->dictSize; i++)
        dict->wordList[i] = (char*)calloc(WORD_LENGTH, sizeof(char));
    return dict;
}

SentenceList* initializeSentenceList() {
    SentenceList *sentlist = (SentenceList*)malloc(sizeof(SentenceList));
    sentlist->listSize = 10;
    sentlist->sentenceCounter = 0;
    sentlist->sentenceList = (char**)malloc(sizeof(char*) * sentlist->listSize);
    int i;
    for (i = 0; i < sentlist->listSize; i++) {
        sentlist->sentenceList[i] = (char*)calloc(SENTENCE_LENGTH, sizeof(char));
    }
    return sentlist;
}

bool expandDict(Dictionary *dict) {
    int oldSize = dict->dictSize;
    dict->dictSize += 50;  // add 50 more spaces
    dict->wordList = realloc(dict->wordList, sizeof(char*) * dict->dictSize);
    int i;
    for (i = oldSize; i < dict->dictSize; i++) {
        dict->wordList[i] = (char*)malloc(sizeof(char) * WORD_LENGTH);
    }
    return dict->wordList ? true : false;
}

bool expandSent(SentenceList *sentl) {
    int oldSize = sentl->listSize;
    sentl->listSize += 50;  // add 50 more spaces
    sentl->sentenceList = realloc(sentl->sentenceList, sizeof(char*) * sentl->listSize);
    int i;
    for (i = oldSize; i < sentl->listSize; i++) {
        sentl->sentenceList[i] = (char*)malloc(sizeof(char) * SENTENCE_LENGTH);
    }
    return sentl->sentenceList ? true : false;
}

bool isExist(char *word, Dictionary *dict) {
    int i = 0;
    while (i < dict->wordCounter) {
        if (!strcmp(dict->wordList[i], word))
            return true;
        i++;
    }
    return false;
}

void printDict(Dictionary *dict) {
    int i;
    system("cls");
    printf("\n\t$ Sports Dictionary $\n");
    for (i = 0; i < dict->wordCounter; i++) {
        printf("\n\t# Word-%02d : %s", i + 1, dict->wordList[i]);
    }
}

void trimWord(char *w) {
    int i, len = strlen(w);
    if (len == 1) {
        w[0] = '\0';
        return;
    }
    for (i = 0; i < len; i++) {
        if (isdigit(w[i])) {
            w[0] = '\0';  // Null the word if it contains any digit
            break;
        }
        if (ispunct(w[i])) {
            if (w[i] != '-') {
                w[i] = '\0';
                break;
            } else if (i > 0 && i < len - 1 && (!isalpha(w[i - 1]) || !isalpha(w[i + 1]))) {
                w[i] = '\0';
                break;
            }
        } else {
            w[i] = tolower(w[i]);  // make alphabetical characters lowercase
        }
    }
}

void createHotVector(Dictionary *dict, char *sentence, double *hotVector) {
    int i, j, k;
    char buffer[WORD_LENGTH];
    int sentenceLen = strlen(sentence);
    for (i = 0; i < sentenceLen; i++) {
        if (isalpha(sentence[i])) {
            j = i;
            if (isalpha(sentence[j + 1])) {
                k = 0;
                while (isalpha(sentence[j])) {
                    buffer[k++] = tolower(sentence[j]);
                    j++;
                }
                buffer[k] = '\0';
                i = j;
                // Labeling: set hotVector index corresponding to the word to 1.0
                for (k = 0; k < dict->wordCounter; k++) {
                    if (!strcmp(dict->wordList[k], buffer)) {
                        hotVector[k] = 1.0;
                        break;
                    }
                }
            }
        }
    }
}

void gradientDescent(double *W, SentenceList *sentl, Dictionary *dict) {
    int i, j, iter;
    double wx = 0;
    double grad;
    double desired;
    
    for (iter = 0; iter < ITER; iter++) {
        // Process first 160 sentences
        for (j = 0; j < 160; j++) {
            desired = (j < 80) ? -1.0 : 1.0;
            double *sentence_hotVector = (double*)calloc(dict->wordCounter, sizeof(double));
            createHotVector(dict, sentl->sentenceList[j], sentence_hotVector);
            
            for (i = 0; i < dict->wordCounter; i++) {
                wx = W[i] * sentence_hotVector[i];
                grad = (1.0 / dict->wordCounter) * (-2 * (desired - tanh(wx)) * dtanh(wx) * sentence_hotVector[i]);
                W[i] = W[i] - LR * grad;
            }
            free(sentence_hotVector);
        }
    }
}

void stochasticGradientDescent(double *W, SentenceList *sentl, Dictionary *dict) {
    int i, j, k, iter;
    double wx = 0;
    double grad;
    double desired;
    
    for (iter = 0; iter < ITER; iter++) {
        // Process in batches
        for (k = 0; k < BATCH_SIZE; k++) {
            j = rand() % 160;
            desired = (j < 80) ? -1.0 : 1.0;
            double *sentence_hotVector = (double*)calloc(dict->wordCounter, sizeof(double));
            createHotVector(dict, sentl->sentenceList[j], sentence_hotVector);
            
            for (i = 0; i < dict->wordCounter; i++) {
                wx = W[i] * sentence_hotVector[i];
                grad = (1.0 / dict->wordCounter) * (-2 * (desired - tanh(wx)) * dtanh(wx) * sentence_hotVector[i]);
                W[i] = W[i] - LR * grad;
            }
            free(sentence_hotVector);
        }
    }
}

double predictdata(double *trained_W, double *testData, int dictSize) {
    int i;
    double result = 0;
    double prediction;
    
    for (i = 0; i < dictSize; i++) {
        result += trained_W[i] * testData[i];
    }
    prediction = tanh(result);
    printf("\nPrediction result: %f ", prediction);
    
    if (prediction > 0.3) {
        printf("Context of Sports Dictionary\n");
    } else if (prediction < -0.3) {
        printf("Context of Politics Dictionary\n");
    } else {
        printf("Success rate too low to make a prediction\n");
    }
    return prediction;
}

void ADAM(double *W, SentenceList *sentl, Dictionary *dict) {
    int i, j, epoch;
    double b1 = 0.9;
    double b2 = 0.999;
    double e = 1e-8;
    double alpha = LR * sqrt(1.0 - pow(b2, 1)) / (1.0 - pow(b1, 1));
    double wx = 0;
    double Vw = 0;
    double Sw = 0;
    double grad;
    double desired;
    
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        for (j = 0; j < 160; j++) {
            desired = (j < 80) ? -1.0 : 1.0;
            double *sentence_hotVector = (double*)calloc(dict->wordCounter, sizeof(double));
            createHotVector(dict, sentl->sentenceList[j], sentence_hotVector);
            
            for (i = 0; i < dict->wordCounter; i++) {
                wx = W[i] * sentence_hotVector[i];
                grad = (1.0 / dict->wordCounter) * (-2 * (desired - tanh(wx)) * dtanh(wx) * sentence_hotVector[i]);
                Vw = b1 * Vw + (1 - b1) * grad;
                Sw = b2 * Sw + (1 - b2) * (grad * grad);
                W[i] = W[i] - alpha * Vw / sqrt(Sw + e);
            }
            free(sentence_hotVector);
        }
    }
}