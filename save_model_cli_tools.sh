#查看tf模型参数 saved_model_cli 命令行工具

model_path=./keras_saved_graph
saved_model_cli show --dir $model_path --all

#说明：
#tag-set类似于版本号信息，signatureDefs表示模型签名
#signature_def['__saved_model_init_op']: 模型初始化
#signature_def['serving_default']: 利用它来提供服务
#模型签名包括：inputs、outputs、Method name


##### 我们使用save_model-cli命令行工具还可以指定不同的tag_set和signature_def ↓↓
saved_model_cli show --dir $model_path \
    --tag_set serve --signature_def serving_default


####我们还可以使用saved_model_cli对模型进行测试
saved_model_cli run --dir ./keras_saved_graph --tag_set serve \
    --signature_def serving_default \
    --input_exprs 'input_1=np.ones((10,200))'