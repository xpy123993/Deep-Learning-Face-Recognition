#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

using namespace std;

const int sample_size_maxn = 450;

const int input_width = 25;
const int input_height = 25;

const int patch_num = 32;
const int patch_size = 5;

const int pool_size = 5;

const int in_width2 = input_width / pool_size;
const int in_height2 = input_height / pool_size;
const int in_width4 = input_width / pool_size / pool_size;
const int in_height4 = input_height / pool_size / pool_size;
const int label_size = 41;

const bool print_patch_data = false;

double learn_rate = 7e-3;

const double inf = 1e7;

char input_images[sample_size_maxn][input_height][input_width];
int input_image_tags[sample_size_maxn];

char test_images[sample_size_maxn][input_height][input_width];
int test_image_tags[sample_size_maxn];

int sample_size = 0;
int test_sample_size = 0;

char input_image[input_height][input_width];
double patch[patch_num][patch_size][patch_size];

//layer1 conv-layer
double layer1_result[patch_num][input_height][input_width];
double layer2_result[patch_num][in_height2][in_width2];
double layer3_result[patch_num][patch_num][in_height2][in_width2];
double layer4_result[patch_num][patch_num][in_height4][in_width4];
double layer5_result[patch_num * patch_num * in_height4 * in_width4];

double softmax_w[patch_num * patch_num * in_height4 * in_width4][label_size];
double softmax_b[label_size];

double softmax_dif[patch_num * patch_num * in_height4 * in_width4];
double pool_layer_2_dif[patch_num][patch_num][in_height2][in_width2];
double pool_layer_1_dif[patch_num][input_height][input_width];

double conv_layer_2_dif[patch_num][in_height2][in_width2];

double final_result[label_size];
double expect_result[label_size];

double rand_double(){ return -1.0 + (rand() % 20000) / 10000.0;}

double err_sum = 0;
int correct_sum = 0;

int debug_count = 0;

void store_environment()
{
    FILE *fp = fopen("state.dat", "wb");
    for(int patch_id = 0; patch_id < patch_num; patch_id ++)
    {
        for(int i = 0; i < patch_size; i ++)
            fwrite(patch[patch_id][i], sizeof(double), patch_size, fp);
    }
    for(int i = 0; i < patch_num * patch_num * in_height4 * in_width4; i ++)
        fwrite(softmax_w[i], sizeof(double), label_size, fp);
    fwrite(softmax_b, sizeof(double), label_size, fp);
    fclose(fp);
}

void restore_environment()
{
    FILE *fp = fopen("state.dat", "rb");
    if(!fp) return;
    for(int patch_id = 0; patch_id < patch_num; patch_id ++)
    {
        if(print_patch_data) printf("Patch #%02d:\n", patch_id);
        for(int i = 0; i < patch_size; i ++)
        {
            fread(patch[patch_id][i], sizeof(double), patch_size, fp);
            if(print_patch_data)
            {
                for(int j = 0; j < patch_size; j ++)
                    printf("%3.2f\t", patch[patch_id][i][j]);
                printf("\n");
            }
        }
    }
    for(int i = 0; i < patch_num * patch_num * in_height4 * in_width4; i ++)
        fread(softmax_w[i], sizeof(double), label_size, fp);
    fread(softmax_b, sizeof(double), label_size, fp);
    fclose(fp);
}

void load_image(const char* filename)
{
    FILE *fp = fopen(filename, "rb");
    for(int i = 0; i < input_height; i ++)
        fread(input_images[sample_size][i], sizeof(char), input_width, fp);
    fclose(fp);
}

void load_test_image(const char* filename)
{
    FILE *fp = fopen(filename, "rb");
    for(int i = 0; i < input_height; i ++)
        fread(test_images[test_sample_size][i], sizeof(char), input_width, fp);
    fclose(fp);
}

void check_patch()
{
    for(int patch_id = 0; patch_id < patch_num; patch_id ++)
    {
        for(int i = 0; i < patch_size; i ++)
        {
            for(int j = 0; j < patch_size; j ++)
            {
                if(patch[patch_id][i][j] < -inf || patch[patch_id][i][j] > inf)
                {
                    printf("(%d, %d) invalid\n", i, j);
                }
            }
        }
    }
}

void output_filter(const char* filename)
{
    FILE* fp = fopen(filename, "wb");
    fwrite(layer5_result, sizeof(double), patch_num * patch_num * in_height4 * in_width4, fp);
    fclose(fp);
}

bool isInImageBound(int cur_y, int cur_x)
{
    return (0 <= cur_x && cur_x < input_width && 0 <= cur_y && cur_y < input_height);
}

void calculate_layer1()
{
    int cur_x, cur_y;
    double err, sum;
    for(int in_y = 0; in_y < input_height; in_y ++)
    {
        for(int in_x = 0; in_x < input_width; in_x ++)
        {
            for(int patch_id = 0; patch_id < patch_num; patch_id ++)
            {
                sum = 0;
                for(int i = 0; i < patch_size; i ++)
                {
                    for(int j = 0; j < patch_size; j ++)
                    {
                        cur_x = in_x + i;
                        cur_y = in_y + j;
                        if(0 <= cur_x && cur_x < input_width && 0 <= cur_y && cur_y < input_height)
                            {

                                sum += patch[patch_id][j][i] * (input_image[cur_y][cur_x] / 256.0);
                                //printf("%f*%d=%f\n", patch[patch_id][j][i], input_image[cur_y][cur_x], patch[patch_id][j][i] * input_image[cur_y][cur_x]);
                            }
                    }
                }
                layer1_result[patch_id][in_y][in_x] = sum;
            }
        }
    }
}



void back_layer1()
{
    int cur_x, cur_y;
    double sum, err;

    for(int in_y = 0; in_y < input_height; in_y ++)
    {
        for(int in_x = 0; in_x < input_width; in_x ++)
        {
            for(int patch_id = 0; patch_id < patch_num; patch_id ++)
            {
                sum = 0;
                err = pool_layer_1_dif[patch_id][in_y][in_x];
                //printf("layer1_err = %f\n", err);

                for(int i = 0; i < patch_size; i ++)
                {
                    for(int j = 0; j < patch_size; j ++)
                    {
                        cur_x = in_x + i;
                        cur_y = in_y + j;
                        if(0 <= cur_x && cur_x < input_width && 0 <= cur_y && cur_y < input_height)
                        {
                            sum += patch[patch_id][j][i];
                            //if(patch[patch_id][j][i] < -inf && debug_count == 1) printf("???:err=%f, value = %f\n, code = %f\n", err, patch[patch_id][j][i], input_image[cur_y][cur_x] / 256.0);
                        }
                    }
                }

                for(int i = 0; i < patch_size; i ++)
                {
                    for(int j = 0; j < patch_size; j ++)
                    {
                        cur_x = in_x + i;
                        cur_y = in_y + j;
                        if(0 <= cur_x && cur_x < input_width && 0 <= cur_y && cur_y < input_height)
                        {
                            bool check = (patch[patch_id][j][i] < -inf || patch[patch_id][j][i] > inf);
                            patch[patch_id][j][i] -= learn_rate * err * (input_image[cur_y][cur_x] / 256.0) / sum;
                            if(check != (patch[patch_id][j][i] < -inf || patch[patch_id][j][i] > inf))
                                printf("layer1-(%d, %d) = %f\n", in_x, in_y, err);
                            //if(patch[patch_id][j][i] < -inf && debug_count == 1) printf("???:err=%f, value = %f\n, code = %f\n", err, patch[patch_id][j][i], input_image[cur_y][cur_x] / 256.0);
                        }
                    }
                }

            }

        }

    }

}

void calculate_layer2()
{
    int cur_x, cur_y;
    double max_value;
    for(int in_y = 0; in_y < input_height; in_y += pool_size)
    {
        for(int in_x = 0; in_x < input_width; in_x += pool_size)
        {
            for(int patch_id = 0; patch_id < patch_num; patch_id ++)
            {
                max_value = layer1_result[patch_id][in_y][in_x];
                for(int i = 0; i < pool_size; i ++)
                {
                    for(int j = 0; j < pool_size; j ++)
                    {
                        cur_y = in_y + i;
                        cur_x = in_x + j;
                        if(isInImageBound(cur_y, cur_x) && max_value < layer1_result[patch_id][cur_y][cur_x])
                            max_value = layer1_result[patch_id][cur_y][cur_x];
                    }
                }
                layer2_result[patch_id][in_y / pool_size][in_x / pool_size] = max_value;
            }
        }
    }
}

void back_layer2()
{
    int cur_x, cur_y, max_y, max_x;
    double max_value;
    for(int in_y = 0; in_y < input_height; in_y += pool_size)
    {
        for(int in_x = 0; in_x < input_width; in_x += pool_size)
        {
            for(int patch_id = 0; patch_id < patch_num; patch_id ++)
            {
                max_value = layer1_result[patch_id][in_y][in_x];
                max_x = in_x;
                max_y = in_y;
                for(int i = 0; i < pool_size; i ++)
                {
                    for(int j = 0; j < pool_size; j ++)
                    {
                        cur_y = in_y + i;
                        cur_x = in_x + j;
                        if(isInImageBound(cur_y, cur_x))
                        {
                            pool_layer_1_dif[patch_id][cur_y][cur_x] = 0;
                            //pool_layer_1_dif[patch_id][cur_y][cur_x] = learn_rate * conv_layer_2_dif[patch_id][cur_y / pool_size][cur_x / pool_size] / pool_size / pool_size;
                            if(max_value < layer1_result[patch_id][cur_y][cur_x] || (max_x == -1 || max_y == -1))
                            {
                                max_value = layer1_result[patch_id][cur_y][cur_x];
                                max_x = cur_x;
                                max_y = cur_y;
                            }
                        }

                    }
                }
                pool_layer_1_dif[patch_id][max_y][max_x] = learn_rate * (conv_layer_2_dif[patch_id][in_y / pool_size][in_x / pool_size]);
                if((conv_layer_2_dif[patch_id][in_y / pool_size][in_x / pool_size] + 1) / pool_size / pool_size < -inf || (conv_layer_2_dif[patch_id][in_y / pool_size][in_x / pool_size] + 1) / pool_size / pool_size > inf)
                    printf("layer2: %f -> %f\n", conv_layer_2_dif[patch_id][in_y / pool_size][in_x / pool_size], (conv_layer_2_dif[patch_id][in_y / pool_size][in_x / pool_size] + 1) / pool_size / pool_size);
            }
        }
    }
}

void calculate_layer3()
{
    int cur_x, cur_y;
    double sum;
    for(int in_y = 0; in_y < input_height / pool_size; in_y ++)
    {
        for(int in_x = 0; in_x < input_width / pool_size; in_x ++)
        {
            for(int patch_id = 0; patch_id < patch_num; patch_id ++)
            {
                for(int patch_id1 = 0; patch_id1 < patch_num; patch_id1 ++)
                {
                    sum = 0;
                    for(int i = 0; i < patch_size; i ++)
                    {
                        for(int j = 0; j < patch_size; j ++)
                        {
                            cur_x = in_x + i;
                            cur_y = in_y + j;
                            if(0 <= cur_x && cur_x < in_width2 &&
                               0 <= cur_y && cur_y < in_height2)
                                sum += patch[patch_id1][j][i] * layer2_result[patch_id][cur_y][cur_x];
                        }
                    }
                    layer3_result[patch_id1][patch_id][in_y][in_x] = sum;
                }
            }
        }
    }
}

void back_layer3()
{
    int cur_x, cur_y;
    double sum, err;

    for(int in_y = 0; in_y < in_height2; in_y ++)
    {
        for(int in_x = 0; in_x < in_width2; in_x ++)
        {
            for(int patch_id = 0; patch_id < patch_num; patch_id ++)
            {
                for(int patch_id1 = 0; patch_id1 < patch_num; patch_id1 ++)
                {
                    sum = 0;
                    err = pool_layer_2_dif[patch_id1][patch_id][in_y][in_x];

                    for(int i = 0; i < patch_size; i ++)
                    {
                        for(int j = 0; j < patch_size; j ++)
                        {
                            cur_x = in_x + i;
                            cur_y = in_y + j;
                            if(0 <= cur_x && cur_x < in_width2 &&
                               0 <= cur_y && cur_y < in_height2)
                                {
                                    sum += patch[patch_id1][j][i];
                                }
                        }
                    }

                    for(int i = 0; i < patch_size; i ++)
                    {
                        for(int j = 0; j < patch_size; j ++)
                        {
                            cur_x = in_x + i;
                            cur_y = in_y + j;
                            if(0 <= cur_x && cur_x < in_width2 &&
                               0 <= cur_y && cur_y < in_height2)
                                {
                                    patch[patch_id1][j][i] -= learn_rate * err * layer2_result[patch_id][cur_y][cur_x] / sum;
                                    conv_layer_2_dif[patch_id1][in_y][in_x] += learn_rate * err * layer2_result[patch_id][cur_y][cur_x] / sum;
                                }
                        }
                    }

                    //printf("%f\n", conv_layer_2_dif[patch_id1][in_y][in_x]);
                    if(conv_layer_2_dif[patch_id1][in_y][in_x] < -inf || conv_layer_2_dif[patch_id1][in_y][in_x] > inf)
                        printf("layer3-crash\n");
                }

            }
        }
    }


}

void calculate_layer4()
{
    int cur_x, cur_y;
    double max_value;
    for(int in_y = 0; in_y < in_height2; in_y += pool_size)
    {
        for(int in_x = 0; in_x < in_width2; in_x += pool_size)
        {
            for(int patch_id = 0; patch_id < patch_num; patch_id ++)
            {
                for(int patch_id1 = 0; patch_id1 < patch_num; patch_id1 ++)
                {
                    max_value = layer3_result[patch_id1][patch_id][in_y][in_x];
                    for(int i = 0; i < pool_size; i ++)
                    {
                        for(int j = 0; j < pool_size; j ++)
                        {
                            cur_y = in_y + i;
                            cur_x = in_x + j;
                            if(0 <= cur_x && cur_x < in_width2 && 0 <= cur_y && cur_y < in_height2 && max_value < layer3_result[patch_id1][patch_id][cur_y][cur_x])
                                max_value = layer3_result[patch_id1][patch_id][cur_y][cur_x];
                        }
                    }
                    layer4_result[patch_id1][patch_id][in_y / pool_size][in_x / pool_size] = max_value;
                }
            }
        }
    }
}

void back_layer4()
{
    int cur_x, cur_y, max_x, max_y, temp_id;
    double max_value;
    for(int in_y = 0; in_y < in_height2; in_y += pool_size)
    {
        for(int in_x = 0; in_x < in_width2; in_x += pool_size)
        {
            for(int patch_id = 0; patch_id < patch_num; patch_id ++)
            {
                for(int patch_id1 = 0; patch_id1 < patch_num; patch_id1 ++)
                {
                    max_x = in_x;
                    max_y = in_y;
                    max_value = layer3_result[patch_id1][patch_id][max_y][max_x];

                    temp_id = (((patch_id1) * patch_num + patch_id) * in_height4 + in_y) * in_width4 + in_x;
                    for(int i = 0; i < pool_size; i ++)
                    {
                        for(int j = 0; j < pool_size; j ++)
                        {
                            cur_y = in_y + i;
                            cur_x = in_x + j;
                            if(0 <= cur_x && cur_x < in_width2 && 0 <= cur_y && cur_y < in_height2)
                            {
                                pool_layer_2_dif[patch_id1][patch_id][cur_y][cur_x] = 0;
                                //pool_layer_2_dif[patch_id1][patch_id][cur_y][cur_x] = learn_rate * softmax_dif[temp_id] / pool_size / pool_size;
                                if(max_value < layer3_result[patch_id1][patch_id][cur_y][cur_x] || (max_x == -1 || max_y == -1))
                                {
                                    max_value = layer3_result[patch_id1][patch_id][cur_y][cur_x];
                                    max_x = cur_x;
                                    max_y = cur_y;
                                }
                            }
                        }
                    }
                    pool_layer_2_dif[patch_id1][patch_id][max_y][max_x] = learn_rate * (softmax_dif[temp_id]) ;
                    if((softmax_dif[temp_id] + 1) / pool_size / pool_size < -inf || (softmax_dif[temp_id] + 1) / pool_size / pool_size > inf)
                        printf("layer5-(%d, %d)\n", in_x, in_y);
                }
            }
        }
    }
}

void calculate_layer5()
{
    int pointer = 0;
    double min_value = inf, max_value = -inf;


    for(int in_y = 0; in_y < in_height4; in_y ++)
    {
        for(int in_x = 0; in_x < in_width4; in_x ++)
        {
            for(int patch_i = 0; patch_i < patch_num; patch_i ++)
            {
                for(int patch_j = 0; patch_j < patch_num; patch_j ++)
                {
                    layer5_result[pointer ++] = layer4_result[patch_j][patch_i][in_y][in_x];

                    min_value = min(min_value, layer4_result[patch_j][patch_i][in_y][in_x]);
                    max_value = max(max_value, layer4_result[patch_j][patch_i][in_y][in_x]);
                }
            }
        }
    }

    if(min_value == max_value) printf("WARNING: NO UNIQUE VALUE FOUND IN LAYER 6, min = max = %f\n", min_value);
    for(int i = 0; i < pointer; i ++)
    {
        layer5_result[i] = layer5_result[i] / (max_value - min_value);
    }
}

void calculate_layer6()
{
    double sum = 1;

    int data_count = patch_num * patch_num * in_height4 * in_width4;
    for(int label_id = 0; label_id < label_size; label_id ++)
    {
        final_result[label_id] = 0;
        for(int i = 0; i < data_count; i ++)
            final_result[label_id] += layer5_result[i] * softmax_w[i][label_id];
        final_result[label_id] += softmax_b[label_id];

        final_result[label_id] = expl(-final_result[label_id]);
        sum += final_result[label_id];

    }
    for(int label_id = 0; label_id < label_size; label_id ++)
    {
        final_result[label_id] = final_result[label_id] / sum;
    }
}

void back_layer6()
{
    double err = 0, sum = 0;
    int data_count = patch_num * patch_num * in_height4 * in_width4;
    err_sum = 0;
    int max_label_id = 0;
    for(int label_id = 0; label_id < label_size; label_id ++)
    {
        if(final_result[max_label_id] < final_result[label_id])
            max_label_id = label_id;
        err = expect_result[label_id] - final_result[label_id];
        err_sum += err < 0 ? -err : err;
        sum = 0;
        for(int i = 0; i < data_count; i ++)
        {
            softmax_w[i][label_id] -= learn_rate * err * layer5_result[i];
            softmax_dif[i] += learn_rate * err * layer5_result[i];
        }
        softmax_b[label_id] -= learn_rate * err;
    }
    if(expect_result[max_label_id] == 1) correct_sum ++;
}

int predict()
{
    int max_index = 0;
    for(int i = 0; i < label_size; i ++)
    {
        if(final_result[i] > final_result[max_index])
            max_index = i;
    }
    return max_index;
}

void initialize_dif_array()
{
    //double softmax_dif[patch_num * patch_num * in_height4 * in_width4];
    //double pool_layer_2_dif[patch_num][patch_num][in_height2][in_width2];
    //double pool_layer_1_dif[patch_num][input_height][input_width];

    //double conv_layer_2_dif[patch_num][in_height2][in_width2];

    int data_count = patch_num * patch_num * in_height4 * in_width4;
    memset(softmax_dif, 0, sizeof(double) * data_count);

    for(int patch_id = 0; patch_id < patch_num; patch_id ++)
    {
        for(int in_y = 0; in_y < input_height; in_y ++)
        {
            for(int in_x = 0; in_x < input_width; in_x ++)
            {
                pool_layer_1_dif[patch_id][in_y][in_x] = 0;
            }
        }
        for(int in_y = 0; in_y < in_height2; in_y ++)
        {
            for(int in_x = 0; in_x < in_width2; in_x ++)
            {
                conv_layer_2_dif[patch_id][in_y][in_x] = 0;
            }
        }
        for(int patch_id1 = 0; patch_id1 < patch_num; patch_id1 ++)
        {
            for(int in_y = 0; in_y < in_height2; in_y ++)
            {
                for(int in_x = 0; in_x < in_width2; in_x ++)
                {
                    pool_layer_2_dif[patch_id1][patch_id][in_y][in_x] = 0;
                }
            }
        }
    }
}

void initialize_patch()
{
    for(int i = 0; i < patch_num; i ++)
    {
        for(int j = 0; j < patch_size * patch_size; j ++)
        {
            patch[i][j / patch_size][j % patch_size] = rand_double();
        }
    }
}

void initialize_softmax()
{
    int data_count = patch_num * patch_num * in_height4 * in_width4;
    for(int label_id = 0; label_id < label_size; label_id ++)
    {
        for(int i = 0; i < data_count; i ++)
        {
            softmax_w[i][label_id] = rand_double();
        }
        softmax_b[label_id] = rand_double();
    }
}

void set_result(int value)
{
    for(int i = 0; i < label_size; i ++)
        expect_result[i] = 0;
    expect_result[value] = 1;
}

void nn_forward()
{
    //printf("forwarding layer1\n");
    calculate_layer1();
    //printf("forwarding layer2\n");
    calculate_layer2();
    //printf("forwarding layer3\n");
    calculate_layer3();
    //printf("forwarding layer4\n");
    calculate_layer4();
    //printf("forwarding layer5\n");
    calculate_layer5();
    //printf("forwarding layer6\n");
    calculate_layer6();

}

void nn_backward()
{
    //printf("backward layer6\n");
    initialize_dif_array();
    back_layer6();
    //printf("backward layer4\n");
    back_layer4();
    //printf("backward layer3\n");
    back_layer3();
    //printf("backward layer2\n");
    back_layer2();
    //printf("backward layer1\n");
    back_layer1();

}

void load_configure(const char* conf_filename)
{
    FILE *fp = fopen(conf_filename, "r");
    char filename[256];
    int tag = 0;

    sample_size = 0;

    while(!feof(fp))
    {
        if(!~fscanf(fp, "%d %s", &tag, filename))
            break;
        input_image_tags[sample_size] = (tag - 1);
        load_image(filename);
        sample_size ++;
    }

    fclose(fp);

    printf("%d sample(s) loaded\n", sample_size);
}

void load_test_configure(const char* conf_filename)
{
    FILE *fp = fopen(conf_filename, "r");
    char filename[256];
    int tag = 0;

    test_sample_size = 0;

    while(!feof(fp))
    {
        if(!~fscanf(fp, "%d %s", &tag, filename))
            break;
        test_image_tags[test_sample_size] = (tag - 1);
        load_test_image(filename);
        test_sample_size ++;
    }

    fclose(fp);

    printf("%d test-sample(s) loaded\n", test_sample_size);
}

void switch_sample(int sample_index)
{
    for(int i = 0; i < input_height; i ++)
        memcpy(input_image[i], input_images[sample_index][i], sizeof(char) * input_width);
    set_result(input_image_tags[sample_index]);
}

void switch_test_sample(int sample_index)
{
    for(int i = 0; i < input_height; i ++)
        memcpy(input_image[i], test_images[sample_index][i], sizeof(char) * input_width);
    set_result(test_image_tags[sample_index]);
}

void train_mode()
{
    int order[sample_size_maxn] = {0}, sample_index;

    double checksum = 0;

    load_configure("configure.txt");
    load_test_configure("configure_test.txt");

    for(int i = 0; i < sample_size; i ++)
    {
        order[i] = i;
    }

    int last_round = 0;



    for(int i = 0; i < 2000; i ++)
    {
        if(i / 100 != last_round)
        {
            last_round = i / 100;
            learn_rate = learn_rate * 0.8;
        }


        random_shuffle(order, order + sample_size);
        for(int order_id = 0; order_id < sample_size; order_id ++)
        {
            sample_index = order[order_id];
            switch_sample(sample_index);

            nn_forward();
            nn_backward();

            checksum += err_sum;

            printf(".");
        }

        FILE *train_file = fopen("train_grid.txt", "a");
        FILE *check_file = fopen("check_grid.txt", "a");

        printf("\nround %d\ntrain-set:\nerrsum = %f, correct rate = %.2f%% (%d/%d)\n", i, checksum,
               100.0 * correct_sum / sample_size, correct_sum, sample_size);

        fprintf(train_file, "%d, %.2f, %.2f\n", i, checksum, 100.0 * correct_sum / sample_size);

        correct_sum = 0;

        for(int test_sample_id = 0; test_sample_id < test_sample_size; test_sample_id ++)
        {
            switch_test_sample(test_sample_id);
            nn_forward();
            //if(expect_result[predict()] == 0) printf("index = %d\n", test_sample_id);
            correct_sum += expect_result[predict()];
            printf(".");
        }
        printf("\ncheck-set:correct rate = %.2f%% (%d/%d)\n", 100.0 * correct_sum / test_sample_size, correct_sum, test_sample_size);

        fprintf(check_file, "%d, %.2f\n", i, 100.0 * correct_sum / test_sample_size);

        correct_sum = 0;
        store_environment();

        if(checksum < 1e-3)
        {
            printf("terminate conditions met\nexit\n");
            store_environment();
            break;
        }
        checksum = 0;

        fclose(train_file);
        fclose(check_file);
    }
}

int main(int argc, char* argv[])
{
    srand(time(0));

    initialize_patch();
    initialize_softmax();


    restore_environment();

    if(argc == 2)
    {
        test_sample_size = 0;
        load_test_image(argv[1]);
        switch_test_sample(0);
        nn_forward();
        printf("识别为编号%d，置信度:%.2f%%\n\n", predict() + 1, 100.0 * final_result[predict()]);
        for(int i = 0; i < label_size; i ++)
        {
            printf("P(type = %02d)=%.2f\n", i, final_result[i]);
        }
    }
    else
        train_mode();


    return 0;
}
