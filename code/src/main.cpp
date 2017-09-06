#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <curl/curl.h>

#include "Image.hpp"
#include "ImageReader.hpp"
#include "MattingPerformer.hpp"
#include "ImagePrinter.hpp"

std::string img_path = "../../back-end/originImage.png";
std::string img_m_path = "../../back-end/scribbleImg.png";

// Global variable
double lambda = 100; // Weight of scribbled piexel obedience
int win_size = 1; // The distance between center and border
double epsilon = 0.00001;
double thresholdForScribble = 0.001;

struct MemoryStruct {
  char *memory;
  size_t size;
};

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
  size_t realsize = size * nmemb;
  struct MemoryStruct *mem = (struct MemoryStruct *)userp;

  mem->memory = (char*)realloc(mem->memory, mem->size + realsize + 1);
  if(mem->memory == NULL) {
    /* out of memory! */
    printf("not enough memory (realloc returned NULL)\n");
    return 0;
  }

  memcpy(&(mem->memory[mem->size]), contents, realsize);
  mem->size += realsize;
  mem->memory[mem->size] = 0;

  return realsize;
}

int performTask()
{
  // Read image
  ImageReader imageReader;
  Image img(imageReader.readImage(img_path));
  Image img_m(imageReader.readImage(img_m_path));
  
  // Perform image matting
  MattingPerformer mattingPerformer(lambda, win_size, epsilon, thresholdForScribble, img.getMatrix(), img_m.getMatrix());
  mattingPerformer.performMatting();
  cv::Mat mattingResultF = mattingPerformer.getMattingResultF();
  cv::Mat mattingResultB = mattingPerformer.getMattingResultB();

  // Print matting result
  Image_Printer imagePrinter;
  imagePrinter.printImage(mattingResultF, "foreground");
  imagePrinter.printImage(mattingResultB, "background");
  
  return 0;
}

void notifyJobIsDone() {
  CURL *curl;
  CURLcode res;

  /* In windows, this will init the winsock stuff */
  curl_global_init(CURL_GLOBAL_ALL);

  /* get a curl handle */
  curl = curl_easy_init();
  if(curl) {
    /* First set the URL that is about to receive our POST. This URL can
       just as well be a https:// URL if that is what should receive the
       data. */
    curl_easy_setopt(curl, CURLOPT_URL, "localhost:3000/api/imgProcessed");
    /* Now specify the POST data */
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "Job is done!");

    /* Perform the request, res will get the return code */
    res = curl_easy_perform(curl);
    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  curl_global_cleanup();
}

int main(void)
{
  CURL *curl_handle;
  CURLcode res;

  while (1) {
    struct MemoryStruct chunk;

    chunk.memory = (char*)malloc(1);  /* will be grown as needed by the realloc above */
    chunk.size = 0;    /* no data at this point */

    curl_global_init(CURL_GLOBAL_ALL);

    /* init the curl session */
    curl_handle = curl_easy_init();

    /* specify URL to get */
    curl_easy_setopt(curl_handle, CURLOPT_URL, "localhost:3000/getJob");

    /* send all data to this function  */
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);

    /* we pass our 'chunk' struct to the callback function */
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);

    /* some servers don't like requests that are made without a user-agent
       field, so we provide one */
    curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");

    /* get it! */
    res = curl_easy_perform(curl_handle);

    /* check for errors */
    if(res != CURLE_OK) {
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));
    }
    else {
      printf("%s\n", chunk.memory);
      if (strcmp(chunk.memory, "Ready") == 0) {
        printf("Job start!\n");
        performTask(); // about 5 seoncd to handle the job
        notifyJobIsDone();
      } else if (strcmp(chunk.memory, "No Data Yet!") == 0){
        printf("Job not available yet, waiting for another 10 seconds\n");
        sleep(10); // 10 seconds
      }
    }

    /* cleanup curl stuff */
    curl_easy_cleanup(curl_handle);

    free(chunk.memory);
  }

  /* we're done with libcurl, so clean it up */
  curl_global_cleanup();

  return 0;
}
