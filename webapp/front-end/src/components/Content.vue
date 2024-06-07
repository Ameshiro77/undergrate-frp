<!-- eslint-disable no-console -->
<template>
  <div id="Content" style="width: 80%; justify-content: center">
    <el-dialog
      title="AI预测中"
      :visible.sync="dialogTableVisible"
      :show-close="false"
      :close-on-press-escape="false"
      :append-to-body="true"
      :close-on-click-modal="false"
      :center="true"
    >
      <el-progress :percentage="percentage"></el-progress>
      <span slot="footer" class="dialog-footer">请耐心等待</span>
    </el-dialog>

    <div id="HOI">
      <!-- HOI IMAGE DIV -->
      <!-- 传图片，显示图片和检测结果 -->
      <div id="HOI-Image" style="display: flex">
        <!-- 上传的图 -->
        <!-- 用card的方式显示 -->
        <el-card
          id="image-card"
          class="box-card"
          style="
            border-radius: 8px;
            width: 100%;
            height: 400px;
            margin-bottom: -30px;
          "
        >
          <div
            id="hoi-detec"
            style="display: flex; justify-content: space-between; width: 100%"
          >
            <!-- 上传的图 -->
            <div>
              <div
                v-loading="loading"
                element-loading-text="上传图片中"
                element-loading-spinner="el-icon-loading"
              >
                <el-image
                  :src="url_input"
                  class="image_src"
                  :preview-src-list="[url_input]"
                  style="border-radius: 3px 3px 0 0"
                >
                  <div slot="error">
                    <div slot="placeholder" class="error">
                      <el-button
                        v-show="showbutton"
                        type="primary"
                        icon="el-icon-upload"
                        class="download_bt"
                        v-on:click="true_upload"
                      >
                        上传图像
                        <input
                          ref="upload"
                          style="display: none"
                          name="file"
                          type="file"
                          @change="upload"
                        />
                      </el-button>
                    </div>
                  </div>
                </el-image>
              </div>
              <div class="img_info_1" style="border-radius: 0 0 5px 5px">
                <span style="color: white; letter-spacing: 6px">原始图像</span>
              </div>
            </div>
            <!-- 上传图部分结束 -->

            <!-- 开始检测 -->

            <!-- 设置参数 -->
            <div>
              <!-- 选择框 -->
              <div
                style="display: flex; flex-direction: column; margin-top: 10%"
              >
                <div>
                  请选择最多检测数量：
                  <el-select v-model="rank" placeholder="请选择检测排名数量">
                    <el-option label="1" value="1"></el-option>
                    <el-option label="2" value="2"></el-option>
                    <el-option label="3" value="3"></el-option>
                    <el-option label="4" value="4"></el-option>
                    <el-option label="5" value="5"></el-option>
                    <el-option label="不限" value="-1"></el-option>
                  </el-select>
                </div>
                <div style="margin-top: 10%">
                  请选择分数阈值：
                  <el-select v-model="threshold" placeholder="请选择分数阈值">
                    <el-option label="1.5" value="1.5"></el-option>
                    <el-option label="1.4" value="1.4"></el-option>
                    <el-option label="1.3" value="1.3"></el-option>
                    <el-option label="1.2" value="1.2"></el-option>
                    <el-option label="1.1" value="1.1"></el-option>
                    <el-option label="1.0" value="1.0"></el-option>
                  </el-select>
                </div>
              </div>

              <!-- 开始检测 -->
              <div
                style="display: flex; justify-content: center; margin-top: 10%"
              >
                <el-button
                  style="width: 100%"
                  type="primary"
                  class="download_bt"
                  @click="detect"
                >
                  开始检测
                </el-button>
              </div>

              <!-- 显示开关 -->
              <div
                style="display: flex; justify-content: center; margin-top: 10%"
              >
                <el-switch
                  v-model="isfull"
                  active-text="显示交互"
                  inactive-text="仅显示边框"
                  @change="changeShow"
                >
                  开关
                </el-switch>
              </div>
            </div>

            <!-- 结果的图 -->
            <div>
              <div
                v-loading="loading"
                element-loading-text="处理中,请耐心等待"
                element-loading-spinner="el-icon-loading"
              >
                <el-image
                  :src="result_url"
                  class="image_res"
                  :preview-src-list="[result_url]"
                  style="border-radius: 3px 3px 0 0"
                >
                  <div slot="error">
                    <div slot="placeholder" class="error">
                      {{ wait_return }}
                    </div>
                  </div>
                </el-image>
                <div class="img_info_1" style="border-radius: 0 0 5px 5px">
                  <span style="color: white; letter-spacing: 4px"
                    >检测结果</span
                  >
                </div>
              </div>
            </div>
          </div>

          <!-- 重新选择图像按钮 -->
          <div class="reupload">
            <el-button
              style="margin-left: 125px; margin-top: 9px; height: 28px"
              v-show="!showbutton"
              type="primary"
              icon="el-icon-upload"
              class="download_bt"
              v-on:click="true_upload2"
            >
              重新选择图像
              <input
                ref="upload2"
                style="display: none"
                name="file"
                type="file"
                @change="upload"
              />
            </el-button>
          </div>
        </el-card>
      </div>
      <!-- HOI IMAGE END -->

      <div style="margin-top: 5%">
        <el-table :data="tableData" border style="width: 100%">
          <el-table-column prop="name" label="人物交互类别"> </el-table-column>
          <el-table-column prop="sbox" label="人边界框"> </el-table-column>
          <el-table-column prop="obox" label="物体边界框"> </el-table-column>
          <el-table-column prop="score" label="置信度"> </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "Content",
  data() {
    return {
      // 预测结果的文件夹
      pred_full_folder: "http://127.0.0.1:8000/static/result_full/", //预测结果的地址
      pred_nohoi_folder: "http://127.0.0.1:8000/static/result_no_hoi/",
      // 具体结果地址
      isfull: true,
      result_url: "",
      result_full_url: "",
      result_nohoi_url: "",
      // param参数：file rank threshold
      file: null,
      rank: 3,
      threshold: 1.5,

      centerDialogVisible: true,
      url_input: "", //上传图片的url
      text: "", //医疗建议
      wait_return: "等待上传",
      wait_upload: "等待上传",
      loading: false,
      showbutton: true,
      percentage: 0,
      dialogTableVisible: false,
      tableData: [],
    };
  },

  created: function () {
    //网页名字
    document.title = "DR检测demo";
  },

  methods: {
    true_upload() {
      this.$refs.upload.click();
    },
    true_upload2() {
      this.$refs.upload2.click();
    },
    true_detect() {
      this.$refs.detect.click();
    },

    // 获得目标文件
    getObjectURL(file) {
      var url = null;
      if (window.createObjcectURL != undefined) {
        url = window.createOjcectURL(file);
      } else if (window.URL != undefined) {
        url = window.URL.createObjectURL(file);
      } else if (window.webkitURL != undefined) {
        url = window.webkitURL.createObjectURL(file);
      }
      return url;
    },
    //上传图片
    upload(e) {
      let file = e.target.files[0]; //拿去文件
      this.file = file;
      this.url_input = this.$options.methods.getObjectURL(file); //url_input：上传的图片所在的地址
      this.showbutton = false;
      //console.log(this.file);
    },

    // 上传文件
    detect(e) {
      this.percentage = 0;
      this.wait_return = "";
      this.wait_upload = "";

      if (this.file == null) {
        alert("没有提交图片！");
        return;
      }
      console.log("开始检测");
      this.dialogTableVisible = true;
      this.loading = true; //开启加载

      let param = new FormData(); //创建form对象
      param.append("file", this.file, this.file.name); //通过append向form对象添加数据
      console.log(this.rank,this.threshold)
      param.append("rank", this.rank );
      param.append("threshold", this.threshold);
      console.log(param)

      var timer = setInterval(() => {
        this.myFunc();
      }, 30);

      let config = {
        headers: { "Content-Type": "multipart/form-data" },
      }; //添加请求头

      axios
        .post("http://127.0.0.1:8000/imgUpload", param, config)
        .then((response) => {
          console.log(response);
          this.percentage = 100;
          clearInterval(timer);
          this.result_full_url =
            this.pred_full_folder + response.data[0]["filename"];
          this.result_nohoi_url =
            this.pred_nohoi_folder + response.data[0]["filename"];
          if (this.isfull) {
            this.result_url = this.result_full_url;
          } else {
            this.result_url = this.result_nohoi_url;
          }
          this.updateTable(response.data[1]);
          console.log(this.result_url);
          this.dialogTableVisible = false;
          this.loading = false;
          this.percentage = 0;
          this.putNotice(); //提示预测成功
        });
    },

    changeShow() {
      console.log("change");
      if (this.isfull) {
        this.result_url = this.result_full_url;
      } else {
        this.result_url = this.result_nohoi_url;
      }
    },

    //更新指标
    updateTable(list) {
      this.tableData = [];
      for (let i = 0; i < list.length; i++) {
          let obj = {};
          obj.name = list[i]['hoi_name']
          obj.sbox = list[i]['sub_bbox']
          obj.obox = list[i]['obj_bbox']
          obj.score = list[i]['score']
          this.tableData.push(obj);
      }
    },
    myFunc() {
      if (this.percentage + 10 < 100) {
        this.percentage = this.percentage + 10;
      } else {
        this.percentage = 100;
      }
    },

    // 用于提示预测成功信息
    putNotice() {
      this.$notify({
        title: "预测成功",
        message: "点击图片可以查看大图",
        duration: 5000,
        type: "success",
      });
    },
  },

  mounted() {
    // eslint-disable-next-line no-console
    console.log("组件挂载完毕");
  },
};
</script>

<style>
.el-button {
  padding: 12px 20px !important;
}
</style>

<style scoped>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

.dialog_info {
  margin: 20px auto;
}

.text {
  font-size: 14px;
}

.item {
  margin-bottom: 18px;
}

.box-card {
  width: 100%;
  height: 200px;
  border-radius: 8px;
  margin-top: -20px;
}

.divider {
  width: 50%;
}

.image_src {
  width: 400px;
  height: 300px;
  background: #ffffff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.image_res {
  width: 400px;
  height: 300px;
  background: #ffffff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.img_info_1 {
  height: 30px;
  width: 400px;
  text-align: center;
  background-color: #63aff5;
  line-height: 30px;
}

.error {
  margin: 100px auto;
  width: 50%;
  padding: 10px;
  text-align: center;
}

.block-sidebar {
  position: fixed;
  display: none;
  left: 50%;
  margin-left: 600px;
  top: 350px;
  width: 60px;
  z-index: 99;
}

.block-sidebar .block-sidebar-item {
  font-size: 50px;
  color: lightblue;
  text-align: center;
  line-height: 50px;
  margin-bottom: 20px;
  cursor: pointer;
  display: block;
}

div {
  display: block;
}

.block-sidebar .block-sidebar-item:hover {
  color: #63aff5;
}

.download_bt {
  padding: 10px 16px !important;
}

#upfile {
  width: 104px;
  height: 45px;
  background-color: #63aff5;
  color: #fff;
  text-align: center;
  line-height: 45px;
  border-radius: 3px;
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.1), 0 2px 2px 0 rgba(0, 0, 0, 0.2);
  color: #fff;
  font-family: "Source Sans Pro", Verdana, sans-serif;
  font-size: 0.875rem;
}

.file {
  width: 200px;
  height: 130px;
  position: absolute;
  left: -20px;
  top: 0;
  z-index: 1;
  -moz-opacity: 0;
  -ms-opacity: 0;
  -webkit-opacity: 0;
  opacity: 0; /*css属性&mdash;&mdash;opcity不透明度，取值0-1*/
  filter: alpha(opacity=0);
  cursor: pointer;
}

#upload {
  position: relative;
  margin: 0px 0px;
}

#Content {
  width: 100%;
  background-color: #ffffff;
  margin: 15px auto;
}

.divider {
  background-color: #eaeaea !important;
  height: 2px !important;
  width: 100%;
  margin-bottom: 50px;
}

.divider_1 {
  background-color: #ffffff;
  height: 2px !important;
  width: 100%;
  margin-bottom: 20px;
  margin: 20px auto;
}

.start_detect {
  width: 100%;
  display: flex;
  justify-content: center;
  height: 10%;
  margin-top: 10%;
}

.steps {
  font-family: "lucida grande", "lucida sans unicode", lucida, helvetica,
    "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
  color: #63aff5;
  text-align: center;
  margin: 15px auto;
  font-size: 20px;
  font-weight: bold;
  text-align: center;
}

.step_1 {
  /*color: #303133 !important;*/
  margin: 20px 26px;
}

#info_patient {
  width: 100%;
  margin-top: 5px;
  justify-content: center;
  display: flex;
}
</style>


