const realfilebtn=document.getElementById("real-file");
const custombtn=document.getElementById("mybutton");
const customTxt=document.getElementById("custom-text")

custombtn.addEventListener("click",function(){
  realfilebtn.click();
});


realfilebtn.addEventListener("change", function() {
  if (realfilebtn.value) {
    customTxt.innerHTML = realfilebtn.value.match(
      /[\/\\]([\w\d\s\.\-\(\)]+)$/
    )[1];
  } else {
    customTxt.innerHTML = "No file chosen, yet.";
  }
});
