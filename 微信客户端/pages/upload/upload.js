// pages/upload/upload.js
const app = getApp()
Page({

  /**
   * 页面的初始数据
   */
  data: {
   
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  },

  /**
   * 上传图片按钮点击事件
   */
  upload: function() {
    //获取图片地址
    var tempFilePaths = app.globalData.tempFilePaths
    //显示进度弹框
    wx.showLoading({
      title: '处理中,请耐心等待',
      mask: true
    })
    wx.uploadFile({
      url: 'https://www.ponma.cn:8086/process',
      filePath: tempFilePaths[0],
      name: 'file',
      success(res) {
        //关闭进度弹框
        wx.hideLoading()
        //保存结果
        const result = res.data
        console.log(result)
        app.globalData.result = result
        //界面跳转
        wx.navigateTo({
          url: '../predict/predict'
        })
      },
      fail() {
        //关闭进度弹框
        wx.hideLoading()
        //显示失败弹框
        wx.showToast({
          title: '处理失败',
          icon: 'none',
          duration: 2000
        })
      }
    })
  }
})