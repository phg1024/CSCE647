function writeEmailAddress() {
    var name = 'phg';
    var at = '@';
    var domainname = 'tamu.edu';
    var addr = name + at + domainname;
    document.write('<a href=mailto:' + addr + '>' + addr + '</a>');
}
