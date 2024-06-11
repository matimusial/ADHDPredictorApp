document.querySelectorAll('.tablink').forEach(function(tablink) {
tablink.addEventListener('click', function(e) {
    e.preventDefault();
    document.querySelectorAll('.tablink').forEach(function(link) {
        link.classList.remove('active');
    });
    document.querySelectorAll('.content').forEach(function(content) {
        content.classList.remove('active');
    });
    tablink.classList.add('active');
    document.getElementById(tablink.getAttribute('data-tab')).classList.add('active');
});
});