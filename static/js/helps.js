function postImg(){
        //执行post请求，识别图片
        jQuery("#billmodeltable").remove(); //清空界面识别结果
        if(imgJson['num']==0)
         {   loadingGif('loadingGif');
             imgJson['num']=1;//防止重复提交
            //alert(imgJson["billModel"]);
             imgJson['ocrFlag']=true;
        jQuery.ajax({
            type: "post",
            url: 'ocr',
            data:JSON.stringify({"image":imgJson["imgString"],
                                        "type":imgJson["billModel"]}),
          success:function(d){
              loadingGif('loadingGif');
              imgJson['num']=0;//防止重复提交
              res = JSON.parse(d);
              //imgJson["result"] = res['words_result'];
              create_table(res);
              imgJson['ocrFlag']=false;
          }
        });}
        
         }
		 
function create_table(json){
      $("#recog_res").html("");
      for(var i=0; i<json['result_word'].length; i++){
          var block = json['result_word'][i];
          if(block.is_table == true){//是表格
              var table = $('<table></table>');
              $("#recog_res").append(table);
              var cells = block['cells'];
              var r = 0;
              var tr = "";
              for(var j = 0; j < cells.length; j++){
                  var td = '<td rowspan="'+(cells[j]['end_row']-cells[j]['start_row']+1)+'" colspan="'+(cells[j]['end_col']-cells[j]['start_col']+1)+'" >'+(cells[j]['text'])+' </td>';
                  if(r == cells[j]['start_row']){
                      tr += td;
                  }else {
                      table.append('<tr>'+tr+'</tr>');
                      r = cells[j]['start_row'];
                      tr = td;
                  }
              }
              if(tr){
                  table.append('<tr>'+tr+'</tr>');
			  }
		  }else{//不是表格
              $("#recog_res").append('<div class="block">'+block.text+ '</div>')
		  }
	  }
}


function loadingGif(loadingGif){
        //加载请求时旋转动态图片
        var imgId=document.getElementById(loadingGif);
        if(imgId.style.display=="block")
        {imgId.style.display="none";}
        else
        {imgId.style.display="block";}}


function resize_im(w,h, scale, max_scale){
    f=parseFloat(scale)/Math.min(h, w);
    if(f*Math.max(h, w)>max_scale){
            f=parseFloat(max_scale)/Math.max(h, w);
    }
    newW = parseInt(w*f);
    newH = parseInt(h*f);
    
    return [newW,newH,f]
}


function FunimgPreview(avatarSlect,avatarPreview,myCanvas) {
                //avatarSlect 上传文件控件
                //avatarPreview 预览图片控件
                jQuery("#"+avatarSlect).change(function () {
                var obj=jQuery("#"+avatarSlect)[0].files[0];
                
                var fr=new FileReader();
                fr.readAsDataURL(obj);
                fr.onload=function () {
                      jQuery("#"+avatarPreview).attr('src',this.result);
                      imgJson.imgString = this.result;
                      var image = new Image();
                      image.onload=function(){
                                      var width = image.width;
                                      var height = image.height;
                                      newWH =resize_im(width,height, 800, 1200);    // jche 图像有缩放
                                      newW = newWH[0];
                                      newH = newWH[1];
                                      imgRate = newWH[2];
                                      imgJson.width = width;
                                      imgJson.height = height;
                                      jQuery("#"+avatarPreview).attr('width',newW);
                                      jQuery("#"+avatarPreview).attr('height',newH);
                                      jQuery("#"+'myCanvas').attr('width',newW);
                                      jQuery("#"+'myCanvas').attr('height',newH);
                          
                                      /*
                                      if(width>height){
                                      jQuery("#"+avatarPreview).attr('width',1600);
                                      jQuery("#"+avatarPreview).attr('height',800);
                                      jQuery("#"+'myCanvas').attr('width',1600);
                                      jQuery("#"+'myCanvas').attr('height',800);
                                      }
                                      else{
                                          jQuery("#"+avatarPreview).attr('width',600);
                                          jQuery("#"+avatarPreview).attr('height',1000);
                                          jQuery("#"+myCanvas).attr('width',600);
                                          jQuery("#"+myCanvas).attr('height',1000);
                                      }
                                      */
                                      };
                      image.src= this.result;
                      //box = {"xmin":0,"ymin":0,"xmax":jQuery("#"+'myCanvas').width(),"ymax":jQuery("#"+'myCanvas').height()};                         //createNewCanvas(this.result,'myCanvas',box);
                      
                  
                postImg();//提交POST请求
                };//fr.onload
                
                })//jQuery("#"+avatarSlect)
 }
    
function getChildDetail(){
  jQuery("#billmodeltable").remove();
  childResult = imgJson["result"];
  createTable(childResult);//新建table
}


  

//根据获取的数据，创建table
  //创建table
function createTable(result){
        //根据获取的数据，创建table
        jQuery("#mytable").empty();
        var jsObject = result;
        imgBoxes=[];
        //var jsObject = [{"name":10,"value":20},{"name":10,"value":20}];
        /*var p = "<h3>耗时:"+timeTake+"秒 ,识别结果为:</h3>";
        var tableString =p+ "<table id='billmodeltable' class='gridtable'><tr><th>序号</th><th>值</th></tr>"
                        
        for(var i=0;i<jsObject.length;i++){
            tableString+="<tr><td><p>"+jsObject[i]["name"]+"</p></td><td><p contenteditable='true'>"+jsObject[i]["text"]+"</p></td></tr>";
            imgBoxes.push(jsObject[i]["box"]);
        }*/
        var tableString = "";
        if ( Number(imgJson["billModel"]) > 0 ) {
            tableString +="<table id='billmodeltable' class='gridtable'><tr><th>字段</th><th>值</th></tr>"
                        
            for(var i in jsObject){
                tableString+="<tr><td><p>"+i+"</p></td><td><p contenteditable='true'>"+jsObject[i]+"</p></td></tr>";
            }
        }
        else {
            tableString +="<table id='billmodeltable' class='gridtable'><tr><th>序号</th><th>值</th></tr>"
                        
            for(var i=0;i<jsObject.length;i++){
                tableString+="<tr><td><p>"+i.toString()+"</p></td><td><p contenteditable='true'>"+jsObject[i]["words"]+"</p></td></tr>";
                imgBoxes.push(jsObject[i]["locale"]);
            }
        }
        tableString+="</table>";
        //jQuery("#mytable").append(p);
        jQuery("#mytable").append(tableString);
    drawRbox(imgBoxes,'myCanvas');  
    
    }
        
   
    
function drawRbox(boxes,canvasId){
       /*canvas上绘制倾斜矩形
       */
       var canvas = document.getElementById(canvasId);
       
       if(canvas.getContext){
           
            var ctx = canvas.getContext("2d");
            ctx.strokeStyle = 'rgba(255,0,0,0.5)';
            ctx.lineWidth = 5; 
            ctx.clearRect(0,0,canvas.width,canvas.height);
            ctx.beginPath(); 
            for(var i=0;i<imgBoxes.length;i++){
                   locale=  imgBoxes[i];//x1,y1,x2,y2,x3,y3,x4,y4
                   minx = locale['left_top'][0]
                   miny = locale['left_top'][1]
                   maxx = locale['right_bottom'][0]
                   maxy  = locale['right_bottom'][1]
                   box=  [minx, miny, maxx, miny, maxx, maxy, minx, maxy]
                   x1=box[0]*imgRate;
                   y1=box[1]*imgRate;
                   x2=box[2]*imgRate;
                   y2=box[3]*imgRate;
                   x3=box[4]*imgRate;
                   y3=box[5]*imgRate;
                   x4=box[6]*imgRate;
                   y4=box[7]*imgRate;
                
                   ctx.moveTo(x1, y1);
                   ctx.lineTo(x2, y2);
                
                   ctx.moveTo(x2, y2);
                   ctx.lineTo(x3, y3);
                   
                   ctx.moveTo(x3, y3);
                   ctx.lineTo(x4, y4);
                   
                   ctx.moveTo(x4, y4);
                   ctx.lineTo(x1, y1);
                ctx.stroke();
               }
            
            ctx.closePath();
           }}  


function cycle(){
    if(Math.random()>0.5){
    var canvas = document.getElementById('myCanvas');
    if(canvas.getContext){
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0,0,canvas.width,canvas.height);
        }
    }
    else{
    if(imgJson['ocrFlag']==false){
        drawRbox(imgBoxes,'myCanvas');
    }}
    
}
