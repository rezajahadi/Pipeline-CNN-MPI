#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <endian.h>
#include <string.h>
#include <mpi.h>
#include "cnn.h"

uint32_t be32toh(uint32_t big_endian) {
uint32_t result;
uint8_t *bytes = (uint8_t *)&big_endian;
result = (uint32_t)bytes[0] << 24;
result |= (uint32_t)bytes[1] << 16;
result |= (uint32_t)bytes[2] << 8;
result |= (uint32_t)bytes[3];
return result;
}


/*  IdxFile
 */
typedef struct _IdxFile
{
    int ndims;
    uint32_t* dims;
    uint8_t* data;
} IdxFile;

#define DEBUG_IDXFILE 0

/* IdxFile_read(fp)
   Reads all the data from given fp.
*/
IdxFile* IdxFile_read(FILE* fp)
{
    /* Read the file header. */
    struct {
        uint16_t magic;
        uint8_t type;
        uint8_t ndims;
        /* big endian */
    } header;
    if (fread(&header, sizeof(header), 1, fp) != 1) return NULL;
#if DEBUG_IDXFILE
    fprintf(stderr, "IdxFile_read: magic=%x, type=%x, ndims=%u\n",
            header.magic, header.type, header.ndims);
#endif
    if (header.magic != 0) return NULL;
    if (header.type != 0x08) return NULL;
    if (header.ndims < 1) return NULL;

    /* Read the dimensions. */
    IdxFile* self = (IdxFile*)calloc(1, sizeof(IdxFile));
    if (self == NULL) return NULL;
    self->ndims = header.ndims;
    self->dims = (uint32_t*)calloc(self->ndims, sizeof(uint32_t));
    if (self->dims == NULL) return NULL;
    
    if (fread(self->dims, sizeof(uint32_t), self->ndims, fp) == self->ndims) {
        uint32_t nbytes = sizeof(uint8_t);
        for (int i = 0; i < self->ndims; i++) {
            /* Fix the byte order. */
            uint32_t size = be32toh(self->dims[i]);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: size[%d]=%u\n", i, size);
#endif
            nbytes *= size;
            self->dims[i] = size;
        }
        /* Read the data. */
        self->data = (uint8_t*) malloc(nbytes);
        if (self->data != NULL) {
            fread(self->data, sizeof(uint8_t), nbytes, fp);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: read: %lu bytes\n", n);
#endif
        }
    }

    return self;
}

/* IdxFile_destroy(self)
   Release the memory.
*/
void IdxFile_destroy(IdxFile* self)
{
    assert (self != NULL);
    if (self->dims != NULL) {
        free(self->dims);
        self->dims = NULL;
    }
    if (self->data != NULL) {
        free(self->data);
        self->data = NULL;
    }
    free(self);
}

/* IdxFile_get1(self, i)
   Get the i-th record of the Idx1 file. (uint8_t)
 */
uint8_t IdxFile_get1(IdxFile* self, int i)
{
    assert (self != NULL);
    assert (self->ndims == 1);
    assert (i < self->dims[0]);
    return self->data[i];
}

/* IdxFile_get3(self, i, out)
   Get the i-th record of the Idx3 file. (matrix of uint8_t)
 */
void IdxFile_get3(IdxFile* self, int i, uint8_t* out)
{
    assert (self != NULL);
    assert (self->ndims == 3);
    assert (i < self->dims[0]);
    size_t n = self->dims[1] * self->dims[2];
    memcpy(out, &self->data[i*n], n);
}


/* main */
int main(int argc, char* argv[])
{

    int id, p;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    double start_time, end_time;

    start_time = MPI_Wtime();
    int ncorrect = 0;
    /* argv[1] = train images */
    /* argv[2] = train labels */
    /* argv[3] = test images */
    /* argv[4] = test labels */
    //if (argc < 4) return 100;

    /* Use a fixed random seed for debugging. */
    srand(0);
    /* Initialize layers. */
    /* Input layer - 1x28x28. */
    Layer* linput = Layer_create_input(1, 28, 28);
    /* Conv1 layer - 16x14x14, 3x3 conv, padding=1, stride=2. */
    /* (14-1)*2+3 < 28+1*2 */
    Layer* lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    /* Conv2 layer - 32x7x7, 3x3 conv, padding=1, stride=2. */
    /* (7-1)*2+3 < 14+1*2 */
    Layer* lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    /* FC1 layer - 200 nodes. */
    Layer* lfull1 = Layer_create_full(lconv2, 200, 0.1);
    /* FC2 layer - 200 nodes. */
    Layer* lfull2 = Layer_create_full(lfull1, 200, 0.1);
    /* Output layer - 10 nodes. */
    Layer* loutput = Layer_create_full(lfull2, 10, 0.1);

    /* Read the training images & labels. */


    /* Training finished. */
    FILE* linputf = fopen("linputf.txt", "r");
    if (linputf == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* lconv1f = fopen("lconv1f.txt", "r");
    if (lconv1f == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* lconv2f = fopen("lconv2f.txt", "r");
    if (lconv2f == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* lfull1f = fopen("lfull1f.txt", "r");
    if (lfull1f == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* lfull2f = fopen("lfull2f.txt", "r");
    if (lfull2f == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    FILE* loutputf = fopen("loutputf.txt", "r");
    if (loutputf == NULL) {
        printf("Error opening file for writing.\n");
        return 1;
    }

    Load_pretrainedValues(linput, linputf);
    Load_pretrainedValues(lconv1, lconv1f);
    Load_pretrainedValues(lconv2, lconv2f);
    Load_pretrainedValues(lfull1, lfull1f);
    Load_pretrainedValues(lfull2, lfull2f);
    Load_pretrainedValues(loutput, loutputf);
    

    /* Read the test images & labels. */
    
    IdxFile* images_test = NULL;
    {
        FILE* fp = fopen("./data/t10k-images-idx3-ubyte", "rb");
        if (fp == NULL) return 111;
        images_test = IdxFile_read(fp);
        if (images_test == NULL) return 111;
        fclose(fp);
    }
    IdxFile* labels_test = NULL;
    {
        FILE* fp = fopen("./data/t10k-labels-idx1-ubyte", "rb");
        if (fp == NULL) return 111;
        labels_test = IdxFile_read(fp);
        if (labels_test == NULL) return 111;
        fclose(fp);
    }

    if ( p % 5 == 0)
    {
    if ( id == 0 || id % 5 == 0 ) // input + conv1
    {
        printf("in cpu %d\n", id);
        int ntests = images_test->dims[0];
	int images_per_series = ntests / p * 5;
	
   	int start_index = id / 5 * images_per_series;
   	int end_index = start_index + images_per_series;
   	//printf("no error at image indexing..\n");
   	if (id == p - 5) 
       	   end_index = 10000;
       
        for (int i = start_index; i < end_index; i++) {
            uint8_t img[28*28];
            double x[28*28];
            IdxFile_get3(images_test, i, img);
            for (int j = 0; j < 28*28; j++) {
                x[j] = img[j]/255.0;
            }
            /* Set the values as the outputs. */
            for (int i = 0; i < linput->nnodes; i++) {
                linput->outputs[i] = x[i];
            }

            /* Start feed forwarding for conv1. */
            Layer_feedForw_conv_withInput(lconv1, linput->outputs);

            MPI_Send(lconv1->outputs, (lconv1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
    }

    else if ( id % 5 == 1)  // conv2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv1->nnodes, sizeof(double));
        int images_per_series = 10000 / p * 5;
       
        int start_index = (id-1) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        if (id == p - 4)   
           end_index = 10000;
	int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for conv2. */
            Layer_feedForw_conv_withInput(lconv2, prevl_output);

            MPI_Send(lconv2->outputs, (lconv2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( id % 5 == 2)  // full1
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv2->nnodes, sizeof(double));
	int images_per_series = 10000 / p * 5;
       
        int start_index = (id-2) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        if (id == p - 3)
           end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull1. */
            Layer_feedForw_full_withInput(lfull1, prevl_output);

            MPI_Send(lfull1->outputs, (lfull1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( id % 5 == 3)  // full2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lfull1->nnodes, sizeof(double));
	int images_per_series = 10000 / p * 5;

        int start_index = (id-3) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        if (id == p - 2)
           end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lfull1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull2. */
            Layer_feedForw_full_withInput(lfull2, prevl_output);

            MPI_Send(lfull2->outputs, (lfull2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( id % 5 == 4)  // output
    {
        printf("in cpu %d\n", id);
        int count = 0;
        int ncorrect_series = 0;
        double* prevl_output = (double*)calloc(lfull2->nnodes, sizeof(double));
	int images_per_series = 10000 / p * 5;

        int start_index = (id-4) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        if (id == p - 1)
           end_index = 10000;
        int image_count = end_index-start_index;

        //while (count<image_count)
	for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lfull2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for last layer. */
            Layer_feedForw_full_withInput(loutput, prevl_output);

            double y[10];
            Layer_getOutputs(loutput, y);

            int label = IdxFile_get1(labels_test, i);
            /* Pick the most probable label. */
            int mj = -1;
            for (int j = 0; j < 10; j++) {
                if (mj < 0 || y[mj] < y[j]) {
                    mj = j;
                }
            }
            if (mj == label) {
                ncorrect_series++;
            }
            // if ((i % 1000) == 0) {
            //     fprintf(stderr, "i=%d\n", i);
            // }
            //count++;
	    ncorrect=ncorrect_series;
        }
        printf("in cpu %d  done\n", id);
        fprintf(stderr, "ntests=%d, ncorrect=%d\n", image_count, ncorrect);
        free(prevl_output);
    }
    }

    if ( p % 5 == 1)
    {
    if ( (id == 0 || id % 5 == 0) && (id != p-1) ) // input + conv1
    {
        printf("in cpu %d\n", id);
        int ntests = images_test->dims[0];
	int images_per_series = ntests / p * 5;
	
   	int start_index = id / 5 * images_per_series;
   	int end_index = start_index + images_per_series;
   	//printf("no error at image indexing..\n");
   	//if (id == p - 5) 
       	   //end_index = 10000;
       
        for (int i = start_index; i < end_index; i++) {
            uint8_t img[28*28];
            double x[28*28];
            IdxFile_get3(images_test, i, img);
            for (int j = 0; j < 28*28; j++) {
                x[j] = img[j]/255.0;
            }
            /* Set the values as the outputs. */
            for (int i = 0; i < linput->nnodes; i++) {
                linput->outputs[i] = x[i];
            }

            /* Start feed forwarding for conv1. */
            Layer_feedForw_conv_withInput(lconv1, linput->outputs);

            MPI_Send(lconv1->outputs, (lconv1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
    }

    else if ( (id % 5 == 1) && ( id != p-1 ))  // conv2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv1->nnodes, sizeof(double));
        int images_per_series = 10000 / p * 5;
       
        int start_index = (id-1) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 4)   
           //end_index = 10000;
	int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for conv2. */
            Layer_feedForw_conv_withInput(lconv2, prevl_output);

            MPI_Send(lconv2->outputs, (lconv2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 2) && ( id != p-1 ))  // full1
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = (id-2) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 3)
           //end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull1. */
            Layer_feedForw_full_withInput(lfull1, prevl_output);

            MPI_Send(lfull1->outputs, (lfull1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 3) && ( id != p-1 ))  // full2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lfull1->nnodes, sizeof(double));
	int images_per_series = 10000 / p * 5;

        int start_index = (id-3) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 2)
           //end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lfull1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull2. */
            Layer_feedForw_full_withInput(lfull2, prevl_output);

            MPI_Send(lfull2->outputs, (lfull2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 4) && ( id != p-1 ))  // output
    {
        printf("in cpu %d\n", id);
        int count = 0;
        int ncorrect_series = 0;
        double* prevl_output = (double*)calloc(lfull2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;

        int start_index = (id-4) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 1)
           //end_index = 10000;
        int image_count = end_index-start_index;

        //while (count<image_count)
	for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lfull2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for last layer. */
            Layer_feedForw_full_withInput(loutput, prevl_output);

            double y[10];
            Layer_getOutputs(loutput, y);

            int label = IdxFile_get1(labels_test, i);
            /* Pick the most probable label. */
            int mj = -1;
            for (int j = 0; j < 10; j++) {
                if (mj < 0 || y[mj] < y[j]) {
                    mj = j;
                }
            }
            if (mj == label) {
                ncorrect_series++;
            }
            // if ((i % 1000) == 0) {
            //     fprintf(stderr, "i=%d\n", i);
            // }
            //count++;
	    ncorrect=ncorrect_series;
        }
        printf("in cpu %d  done\n", id);
        fprintf(stderr, "ntests=%d, ncorrect=%d\n", image_count, ncorrect);
        free(prevl_output);
    }
    else if(id == p-1) 
    {
        printf("in cpu %d\n", id);
	int images_per_series = 10000 / p * 5;
	int ncorrect_series =0;
	
   	int start_index = id / 5 * images_per_series; //remaining images;
   	int end_index = 10000;
	int image_count = end_index - start_index;
   	//printf("no error at image indexing..\n");
   	//if (id == p - 5) 
       	   //end_index = 10000;
       
        for (int i = start_index; i < end_index; i++) {
            uint8_t img[28*28];
            double x[28*28];
            IdxFile_get3(images_test, i, img);
            for (int j = 0; j < 28*28; j++) {
                x[j] = img[j]/255.0;
            }
            /* Set the values as the outputs. */
            for (int i = 0; i < linput->nnodes; i++) {
                linput->outputs[i] = x[i];
            }

            /* Start feed forwarding for conv1. */
            Layer_feedForw_conv_withInput(lconv1, linput->outputs);
            Layer_feedForw_conv_withInput(lconv2,lconv1->outputs);
            Layer_feedForw_full_withInput(lfull1, lconv2->outputs);
	    Layer_feedForw_full_withInput(lfull2, lfull1->outputs);
            Layer_feedForw_full_withInput(loutput, lfull2->outputs);
            double y[10];
            Layer_getOutputs(loutput, y);

            int label = IdxFile_get1(labels_test, i);
            /* Pick the most probable label. */
            int mj = -1;
            for (int j = 0; j < 10; j++) {
                if (mj < 0 || y[mj] < y[j]) {
                    mj = j;
                }
            }
            if (mj == label) {
                ncorrect_series++;
            }
            // if ((i % 1000) == 0) {
            //     fprintf(stderr, "i=%d\n", i);
            // }
            //count++;
	    ncorrect=ncorrect_series;
        }
        printf("in cpu %d  done\n", id);
        fprintf(stderr, "ntests=%d, ncorrect=%d\n", image_count, ncorrect);
        //free(prevl_output);
    }

    
    }

    if ( p % 5 == 2)
    {
    if ( (id == 0 || id % 5 == 0) && (id < p-2) ) // input + conv1
    {
        printf("in cpu %d\n", id);
        int ntests = images_test->dims[0];
	int images_per_series = ntests / p * 5;
	
   	int start_index = id / 5 * images_per_series;
   	int end_index = start_index + images_per_series;
   	//printf("no error at image indexing..\n");
   	//if (id == p - 5) 
       	   //end_index = 10000;
       
        for (int i = start_index; i < end_index; i++) {
            uint8_t img[28*28];
            double x[28*28];
            IdxFile_get3(images_test, i, img);
            for (int j = 0; j < 28*28; j++) {
                x[j] = img[j]/255.0;
            }
            /* Set the values as the outputs. */
            for (int i = 0; i < linput->nnodes; i++) {
                linput->outputs[i] = x[i];
            }

            /* Start feed forwarding for conv1. */
            Layer_feedForw_conv_withInput(lconv1, linput->outputs);

            MPI_Send(lconv1->outputs, (lconv1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
    }

    else if ( (id % 5 == 1) && ( id < p-2 ))  // conv2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv1->nnodes, sizeof(double));
        int images_per_series = 10000 / p * 5;
       
        int start_index = (id-1) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 4)   
           //end_index = 10000;
	int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for conv2. */
            Layer_feedForw_conv_withInput(lconv2, prevl_output);

            MPI_Send(lconv2->outputs, (lconv2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 2) && ( id < p-2 ))  // full1
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = (id-2) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 3)
           //end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull1. */
            Layer_feedForw_full_withInput(lfull1, prevl_output);

            MPI_Send(lfull1->outputs, (lfull1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 3) && ( id < p-2 ))  // full2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lfull1->nnodes, sizeof(double));
	int images_per_series = 10000 / p * 5;

        int start_index = (id-3) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 2)
           //end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lfull1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull2. */
            Layer_feedForw_full_withInput(lfull2, prevl_output);

            MPI_Send(lfull2->outputs, (lfull2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 4) && ( id < p-2 ))  // output
    {
        printf("in cpu %d\n", id);
        int count = 0;
        int ncorrect_series = 0;
        double* prevl_output = (double*)calloc(lfull2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;

        int start_index = (id-4) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 1)
           //end_index = 10000;
        int image_count = end_index-start_index;

        //while (count<image_count)
	for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lfull2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for last layer. */
            Layer_feedForw_full_withInput(loutput, prevl_output);

            double y[10];
            Layer_getOutputs(loutput, y);

            int label = IdxFile_get1(labels_test, i);
            /* Pick the most probable label. */
            int mj = -1;
            for (int j = 0; j < 10; j++) {
                if (mj < 0 || y[mj] < y[j]) {
                    mj = j;
                }
            }
            if (mj == label) {
                ncorrect_series++;
            }
            // if ((i % 1000) == 0) {
            //     fprintf(stderr, "i=%d\n", i);
            // }
            //count++;
	    ncorrect=ncorrect_series;
        }
        printf("in cpu %d  done\n", id);
        fprintf(stderr, "ntests=%d, ncorrect=%d\n", image_count, ncorrect);
        free(prevl_output);
    }
    else if(id == p-2) 
    {
        printf("in cpu %d\n", id);
	int images_per_series = 10000 / p * 5;
	int ncorrect_series =0;
	
   	int start_index = id / 5 * images_per_series; //remaining images;
   	int end_index = 10000;
	int image_count = end_index - start_index;
   	//printf("no error at image indexing..\n");
   	//if (id == p - 5) 
       	   //end_index = 10000;
       
        for (int i = start_index; i < end_index; i++) {
            uint8_t img[28*28];
            double x[28*28];
            IdxFile_get3(images_test, i, img);
            for (int j = 0; j < 28*28; j++) {
                x[j] = img[j]/255.0;
            }
            /* Set the values as the outputs. */
            for (int i = 0; i < linput->nnodes; i++) {
                linput->outputs[i] = x[i];
            }

            /* Start feed forwarding for conv1. */
            Layer_feedForw_conv_withInput(lconv1, linput->outputs);
            Layer_feedForw_conv_withInput(lconv2,lconv1->outputs);
            MPI_Send(lconv2->outputs, (lconv2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
    }

    else if (id ==p-1)
    {
        printf("in cpu %d\n", id);
        int count = 0;
	int ncorrect_series=0;
        double* prevl_output = (double*)calloc(lconv2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = id / 5 * images_per_series; //remaining images;
   	int end_index = 10000;
	int image_count = end_index - start_index;

        for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lconv2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull1. */
            Layer_feedForw_full_withInput(lfull1, prevl_output);
	    Layer_feedForw_full_withInput(lfull2, lfull1->outputs);
            Layer_feedForw_full_withInput(loutput, lfull2->outputs);
            double y[10];
            Layer_getOutputs(loutput, y);

            int label = IdxFile_get1(labels_test, i);
            /* Pick the most probable label. */
            int mj = -1;
            for (int j = 0; j < 10; j++) {
                if (mj < 0 || y[mj] < y[j]) {
                    mj = j;
                }
            }
            if (mj == label) {
                ncorrect_series++;
            }
            // if ((i % 1000) == 0) {
            //     fprintf(stderr, "i=%d\n", i);
            // }
            //count++;
	    ncorrect=ncorrect_series;
        }
        printf("in cpu %d done\n", id);
        fprintf(stderr, "ntests=%d, ncorrect=%d\n", image_count, ncorrect);
        free(prevl_output);
    }
    
    }
     if ( p % 5 == 3)
    {
    if ( (id == 0 || id % 5 == 0) && (id < p-3) ) // input + conv1
    {
        printf("in cpu %d\n", id);
        int ntests = images_test->dims[0];
	int images_per_series = ntests / p * 5;
	
   	int start_index = id / 5 * images_per_series;
   	int end_index = start_index + images_per_series;
   	//printf("no error at image indexing..\n");
   	//if (id == p - 5) 
       	   //end_index = 10000;
       
        for (int i = start_index; i < end_index; i++) {
            uint8_t img[28*28];
            double x[28*28];
            IdxFile_get3(images_test, i, img);
            for (int j = 0; j < 28*28; j++) {
                x[j] = img[j]/255.0;
            }
            /* Set the values as the outputs. */
            for (int i = 0; i < linput->nnodes; i++) {
                linput->outputs[i] = x[i];
            }

            /* Start feed forwarding for conv1. */
            Layer_feedForw_conv_withInput(lconv1, linput->outputs);

            MPI_Send(lconv1->outputs, (lconv1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
    }

    else if ( (id % 5 == 1) && ( id < p-3 ))  // conv2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv1->nnodes, sizeof(double));
        int images_per_series = 10000 / p * 5;
       
        int start_index = (id-1) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 4)   
           //end_index = 10000;
	int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for conv2. */
            Layer_feedForw_conv_withInput(lconv2, prevl_output);

            MPI_Send(lconv2->outputs, (lconv2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 2) && ( id < p-3 ))  // full1
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = (id-2) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 3)
           //end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull1. */
            Layer_feedForw_full_withInput(lfull1, prevl_output);

            MPI_Send(lfull1->outputs, (lfull1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 3) && ( id < p-3 ))  // full2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lfull1->nnodes, sizeof(double));
	int images_per_series = 10000 / p * 5;

        int start_index = (id-3) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 2)
           //end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lfull1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull2. */
            Layer_feedForw_full_withInput(lfull2, prevl_output);

            MPI_Send(lfull2->outputs, (lfull2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 4) && ( id < p-3 ))  // output
    {
        printf("in cpu %d\n", id);
        int count = 0;
        int ncorrect_series = 0;
        double* prevl_output = (double*)calloc(lfull2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;

        int start_index = (id-4) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 1)
           //end_index = 10000;
        int image_count = end_index-start_index;

        //while (count<image_count)
	for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lfull2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for last layer. */
            Layer_feedForw_full_withInput(loutput, prevl_output);

            double y[10];
            Layer_getOutputs(loutput, y);

            int label = IdxFile_get1(labels_test, i);
            /* Pick the most probable label. */
            int mj = -1;
            for (int j = 0; j < 10; j++) {
                if (mj < 0 || y[mj] < y[j]) {
                    mj = j;
                }
            }
            if (mj == label) {
                ncorrect_series++;
            }
            // if ((i % 1000) == 0) {
            //     fprintf(stderr, "i=%d\n", i);
            // }
            //count++;
	    ncorrect=ncorrect_series;
        }
        printf("in cpu %d  done\n", id);
        fprintf(stderr, "ntests=%d, ncorrect=%d\n", image_count, ncorrect);
        free(prevl_output);
    }
    else if(id == p-3) 
    {
    printf("in cpu %d\n", id);
	int images_per_series = 10000 / p * 5;
	int ncorrect_series =0;
	
   	int start_index = id / 5 * images_per_series; //remaining images;
   	int end_index = 10000;
	int image_count = end_index - start_index;
   	//printf("no error at image indexing..\n");
   	//if (id == p - 5) 
       	   //end_index = 10000;
       
        for (int i = start_index; i < end_index; i++) {
            uint8_t img[28*28];
            double x[28*28];
            IdxFile_get3(images_test, i, img);
            for (int j = 0; j < 28*28; j++) {
                x[j] = img[j]/255.0;
            }
            /* Set the values as the outputs. */
            for (int i = 0; i < linput->nnodes; i++) {
                linput->outputs[i] = x[i];
            }

            /* Start feed forwarding for conv1. */
            Layer_feedForw_conv_withInput(lconv1, linput->outputs);

            MPI_Send(lconv1->outputs, (lconv1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
    }

    else if (id ==p-2)
    {
        printf("in cpu %d\n", id);
        int count = 0;
	    int ncorrect_series=0;
        double* prevl_output = (double*)calloc(lconv1->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = (id-1) / 5 * images_per_series; //remaining images;
   	    int end_index = 10000;
	    int image_count = end_index - start_index;

        for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lconv1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull1. */
            Layer_feedForw_conv_withInput(lconv2,prevl_output);
            Layer_feedForw_full_withInput(lfull1, lconv2->outputs);
            MPI_Send(lfull1->outputs, (lfull1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if (id == p-1)
    {
        printf("in cpu %d\n", id);
        int count = 0;
	    int ncorrect_series=0;
            double* prevl_output = (double*)calloc(lfull1->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = (id-2) / 5 * images_per_series; //remaining images;
   	    int end_index = 10000;
	    int image_count = end_index - start_index;

        for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lfull1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
	    Layer_feedForw_full_withInput(lfull2, prevl_output);
            Layer_feedForw_full_withInput(loutput, lfull2->outputs);
            double y[10];
            Layer_getOutputs(loutput, y);

            int label = IdxFile_get1(labels_test, i);
            /* Pick the most probable label. */
            int mj = -1;
            for (int j = 0; j < 10; j++) {
                if (mj < 0 || y[mj] < y[j]) {
                    mj = j;
                }
            }
            if (mj == label) {
                ncorrect_series++;
            }
            // if ((i % 1000) == 0) {
            //     fprintf(stderr, "i=%d\n", i);
            // }
            //count++;
	    ncorrect=ncorrect_series;
        }
        printf("in cpu %d done\n", id);
        fprintf(stderr, "ntests=%d, ncorrect=%d\n", image_count, ncorrect);
        free(prevl_output);
    }
    
    }
      if ( p % 5 == 4)
    {
    if ( (id == 0 || id % 5 == 0) && (id < p-4) ) // input + conv1
    {
        printf("in cpu %d\n", id);
        int ntests = images_test->dims[0];
	int images_per_series = ntests / p * 5;
	
   	int start_index = id / 5 * images_per_series;
   	int end_index = start_index + images_per_series;
   	//printf("no error at image indexing..\n");
   	//if (id == p - 5) 
       	   //end_index = 10000;
       
        for (int i = start_index; i < end_index; i++) {
            uint8_t img[28*28];
            double x[28*28];
            IdxFile_get3(images_test, i, img);
            for (int j = 0; j < 28*28; j++) {
                x[j] = img[j]/255.0;
            }
            /* Set the values as the outputs. */
            for (int i = 0; i < linput->nnodes; i++) {
                linput->outputs[i] = x[i];
            }

            /* Start feed forwarding for conv1. */
            Layer_feedForw_conv_withInput(lconv1, linput->outputs);

            MPI_Send(lconv1->outputs, (lconv1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
    }

    else if ( (id % 5 == 1) && ( id < p-4 ))  // conv2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv1->nnodes, sizeof(double));
        int images_per_series = 10000 / p * 5;
       
        int start_index = (id-1) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 4)   
           //end_index = 10000;
	int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for conv2. */
            Layer_feedForw_conv_withInput(lconv2, prevl_output);

            MPI_Send(lconv2->outputs, (lconv2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 2) && ( id < p-4 ))  // full1
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lconv2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = (id-2) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 3)
           //end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lconv2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull1. */
            Layer_feedForw_full_withInput(lfull1, prevl_output);

            MPI_Send(lfull1->outputs, (lfull1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 3) && ( id < p-4 ))  // full2
    {
        printf("in cpu %d\n", id);
        int count = 0;
        double* prevl_output = (double*)calloc(lfull1->nnodes, sizeof(double));
	int images_per_series = 10000 / p * 5;

        int start_index = (id-3) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 2)
           //end_index = 10000;
        int image_count = end_index-start_index;

        while (count<image_count)
        {
            MPI_Recv(prevl_output, ((lfull1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull2. */
            Layer_feedForw_full_withInput(lfull2, prevl_output);

            MPI_Send(lfull2->outputs, (lfull2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
            count++;
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if ( (id % 5 == 4) && ( id < p-4 ))  // output
    {
        printf("in cpu %d\n", id);
        int count = 0;
        int ncorrect_series = 0;
        double* prevl_output = (double*)calloc(lfull2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;

        int start_index = (id-4) / 5 * images_per_series;
        int end_index = start_index + images_per_series;
        //printf("no error at image indexing..\n");
        //if (id == p - 1)
           //end_index = 10000;
        int image_count = end_index-start_index;

        //while (count<image_count)
	for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lfull2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for last layer. */
            Layer_feedForw_full_withInput(loutput, prevl_output);

            double y[10];
            Layer_getOutputs(loutput, y);

            int label = IdxFile_get1(labels_test, i);
            /* Pick the most probable label. */
            int mj = -1;
            for (int j = 0; j < 10; j++) {
                if (mj < 0 || y[mj] < y[j]) {
                    mj = j;
                }
            }
            if (mj == label) {
                ncorrect_series++;
            }
            // if ((i % 1000) == 0) {
            //     fprintf(stderr, "i=%d\n", i);
            // }
            //count++;
	    ncorrect=ncorrect_series;
        }
        printf("in cpu %d  done\n", id);
        fprintf(stderr, "ntests=%d, ncorrect=%d\n", image_count, ncorrect);
        free(prevl_output);
    }
    else if(id == p-4) 
    {
    printf("in cpu %d\n", id);
	int images_per_series = 10000 / p * 5;
	int ncorrect_series =0;
	
   	int start_index = id / 5 * images_per_series; //remaining images;
   	int end_index = 10000;
	int image_count = end_index - start_index;
   	//printf("no error at image indexing..\n");
   	//if (id == p - 5) 
       	   //end_index = 10000;
       
        for (int i = start_index; i < end_index; i++) {
            uint8_t img[28*28];
            double x[28*28];
            IdxFile_get3(images_test, i, img);
            for (int j = 0; j < 28*28; j++) {
                x[j] = img[j]/255.0;
            }
            /* Set the values as the outputs. */
            for (int i = 0; i < linput->nnodes; i++) {
                linput->outputs[i] = x[i];
            }

            /* Start feed forwarding for conv1. */
            Layer_feedForw_conv_withInput(lconv1, linput->outputs);

            MPI_Send(lconv1->outputs, (lconv1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
    }

    else if (id ==p-3)
    {
        printf("in cpu %d\n", id);
        int count = 0;
	    int ncorrect_series=0;
        double* prevl_output = (double*)calloc(lconv1->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = (id-1) / 5 * images_per_series; //remaining images;
   	    int end_index = 10000;
	    int image_count = end_index - start_index;

        for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lconv1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull1. */
            Layer_feedForw_conv_withInput(lconv2,prevl_output);
            MPI_Send(lconv2->outputs, (lconv2->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }
    else if (id ==p-2)
    {
        printf("in cpu %d\n", id);
        int count = 0;
	    int ncorrect_series=0;
        double* prevl_output = (double*)calloc(lconv2->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = (id-1) / 5 * images_per_series; //remaining images;
   	    int end_index = 10000;
	    int image_count = end_index - start_index;

        for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lconv2->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
            /* Start feed forwarding for lfull1. */
            Layer_feedForw_full_withInput(lfull1, prevl_output);
            MPI_Send(lfull1->outputs, (lfull1->nnodes), MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
        }
        printf("in cpu %d done\n", id);
        free(prevl_output);
    }

    else if (id == p-1)
    {
        printf("in cpu %d\n", id);
        int count = 0;
	    int ncorrect_series=0;
            double* prevl_output = (double*)calloc(lfull1->nnodes, sizeof(double));
	    int images_per_series = 10000 / p * 5;
       
        int start_index = (id-2) / 5 * images_per_series; //remaining images;
   	    int end_index = 10000;
	    int image_count = end_index - start_index;

        for (int i = start_index; i < end_index; i++) 
        {
            MPI_Recv(prevl_output, ((lfull1->nnodes)), MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
	        Layer_feedForw_full_withInput(lfull2, prevl_output);
            Layer_feedForw_full_withInput(loutput, lfull2->outputs);
            double y[10];
            Layer_getOutputs(loutput, y);

            int label = IdxFile_get1(labels_test, i);
            /* Pick the most probable label. */
            int mj = -1;
            for (int j = 0; j < 10; j++) {
                if (mj < 0 || y[mj] < y[j]) {
                    mj = j;
                }
            }
            if (mj == label) {
                ncorrect_series++;
            }
            // if ((i % 1000) == 0) {
            //     fprintf(stderr, "i=%d\n", i);
            // }
            //count++;
	    ncorrect=ncorrect_series;
        }
        printf("in cpu %d done\n", id);
        fprintf(stderr, "ntests=%d, ncorrect=%d\n", image_count, ncorrect);
        free(prevl_output);
    }
    
    }
    // Reduce ncorrect across all processes
    int total_correct;
    MPI_Reduce(&ncorrect, &total_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    double execution_time = end_time - start_time;

    if (id == 0) {
        printf("Total correct predictions: %d\n", total_correct);
	printf("Total execution time: %f seconds\n", execution_time);
    }


    IdxFile_destroy(images_test);
    IdxFile_destroy(labels_test);

    Layer_destroy(linput);
    Layer_destroy(lconv1);
    Layer_destroy(lconv2);
    Layer_destroy(lfull1);
    Layer_destroy(lfull2);
    Layer_destroy(loutput);

    // Close file
    fclose(linputf);
    fclose(lconv1f);
    fclose(lconv2f);
    fclose(lfull1f);
    fclose(lfull2f);
    fclose(loutputf);

    MPI_Finalize();
    return 0;
}