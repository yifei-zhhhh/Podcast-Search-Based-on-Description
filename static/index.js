$(document).ready(function(){
        $('#button').click(function()
    {
        $.post("/index",
            {

                "sentence":$("#input_sentence").val(),
            },
        function(data, status)
        {
            // alert(data.sentence);
            $("#output_sentence").text(data.sentence)
        });
    });
})