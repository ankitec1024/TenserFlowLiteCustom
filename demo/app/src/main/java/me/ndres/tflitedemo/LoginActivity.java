package me.ndres.tflitedemo;


import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import java.util.HashMap;
import java.util.Map;


public class LoginActivity extends Activity {
    private Button btnLogin;
    private EditText etUsername,etPassword;
    private TextView tvCreateAccount;
    SharedPreferences login;
    SharedPreferences.Editor loginedit;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        login = getSharedPreferences("NIGHT_MODE",MODE_PRIVATE);
        loginedit = login.edit();

        setContentView(R.layout.activity_login);
        initViews();
    }

    private void initViews(){
        btnLogin= (Button) findViewById(R.id.btnLogin);
        etUsername= (EditText) findViewById(R.id.etUserName);
        etPassword= (EditText) findViewById(R.id.etPassword);
        tvCreateAccount= (TextView) findViewById(R.id.tvCreateAccount);

        btnLogin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(Validation()){

                    loginUser();
                }
            }

        });

        tvCreateAccount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Intent intent = new Intent(LoginActivity.this, LoginActivity.class);
                startActivity(intent);
            }
        });
    }

    private boolean Validation() {

        if(etUsername.getText().toString().length()==0){
            etUsername.setError("Enter username");
            return false;
        }
        else if(etPassword.getText().toString().length()==0){
            etPassword.setError("Enter password");
            return false;
        }
        else
            return  true;

    }

    private void loginUser(){

        StringRequest stringRequest = new StringRequest(Request.Method.POST, "https://mvp.verify24x7.in/emotional/api/tasks/saveEmotions",
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {

                        Toast.makeText(getApplicationContext(), "Login Success", Toast.LENGTH_LONG).show();
                        loginedit.putString("name", etUsername.getText().toString()).commit();

                        Intent i = new Intent(getApplicationContext(), CameraActivity.class);
                        startActivity(i);
                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Toast.makeText(getApplicationContext(), "wrong", Toast.LENGTH_LONG).show();
                    }
                }) {
            @Override
            protected Map<String, String> getParams() {
                Map<String, String> map = new HashMap<>();
                map.put("username", etUsername.getText().toString());
                map.put("password", etPassword.getText().toString());
                return map;
            }
        };

        RequestQueue requestQueue = Volley.newRequestQueue(this);
        stringRequest.setRetryPolicy(new DefaultRetryPolicy(0, DefaultRetryPolicy.DEFAULT_MAX_RETRIES, 2));

        requestQueue.add(stringRequest);


    }

}
