function writeEmailAddress() {
    var name = 'phg';
    var at = '@';
    var domainname = 'tamu.edu';
    var addr = name + at + domainname;
    document.write('<a class="mylink" href=mailto:' + addr + '>' + addr + '</a>');
}

$(document).ready(function(){
    $('.fbox').fancybox({
        helpers: {
            title : {
                type : 'float'
            }
        }
    });
});